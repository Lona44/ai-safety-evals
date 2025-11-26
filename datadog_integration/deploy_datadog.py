#!/usr/bin/env python3
"""
Deploy Datadog Dashboard and Monitors for AI Safety Evals.

This script uses the Datadog API to create/update:
1. The AI Safety Evals dashboard
2. Detection rules (monitors) for misalignment alerts

Usage:
    python datadog_integration/deploy_datadog.py

Requirements:
    - DD_API_KEY environment variable
    - DD_APP_KEY environment variable (for creating dashboards/monitors)
    - DD_SITE environment variable (default: ap2.datadoghq.com)
"""

import json
import os
import sys
from pathlib import Path

try:
    from datadog_api_client import ApiClient, Configuration
    from datadog_api_client.v1.api.dashboards_api import DashboardsApi
    from datadog_api_client.v1.api.monitors_api import MonitorsApi
except ImportError:
    print("Error: datadog-api-client not installed.")
    print("Install with: pip install datadog-api-client")
    sys.exit(1)


def get_config():
    """Get Datadog API configuration from environment."""
    api_key = os.getenv("DD_API_KEY")
    app_key = os.getenv("DD_APP_KEY")
    site = os.getenv("DD_SITE", "ap2.datadoghq.com")

    if not api_key:
        print("Error: DD_API_KEY environment variable not set")
        sys.exit(1)

    if not app_key:
        print("Error: DD_APP_KEY environment variable not set")
        print(
            "Create an application key at: https://app.datadoghq.com/organization-settings/application-keys"
        )
        sys.exit(1)

    configuration = Configuration()
    configuration.api_key["apiKeyAuth"] = api_key
    configuration.api_key["appKeyAuth"] = app_key
    configuration.server_variables["site"] = site

    return configuration


def deploy_dashboard(config: Configuration):
    """Deploy the AI Safety Evals dashboard."""
    dashboard_file = Path(__file__).parent / "dashboard.json"

    if not dashboard_file.exists():
        print(f"Error: Dashboard file not found: {dashboard_file}")
        return None

    with open(dashboard_file) as f:
        dashboard_def = json.load(f)

    with ApiClient(config) as api_client:
        api = DashboardsApi(api_client)

        # Check if dashboard already exists
        try:
            dashboards = api.list_dashboards()
            existing = next(
                (d for d in dashboards.dashboards if d.title == dashboard_def["title"]),
                None,
            )

            if existing:
                print(f"Updating existing dashboard: {existing.id}")
                response = api.update_dashboard(existing.id, dashboard_def)
                print(f"Dashboard updated: {response.url}")
                return response.id
            else:
                print("Creating new dashboard...")
                response = api.create_dashboard(dashboard_def)
                print(f"Dashboard created: {response.url}")
                return response.id

        except Exception as e:
            print(f"Error deploying dashboard: {e}")
            return None


def deploy_monitors(config: Configuration):
    """Deploy AI Safety detection rule monitors."""
    monitors_file = Path(__file__).parent / "monitors.json"

    if not monitors_file.exists():
        print(f"Error: Monitors file not found: {monitors_file}")
        return []

    with open(monitors_file) as f:
        monitors_def = json.load(f)

    deployed = []

    with ApiClient(config) as api_client:
        api = MonitorsApi(api_client)

        # Get existing monitors
        try:
            existing_monitors = api.list_monitors()
            existing_names = {m.name: m.id for m in existing_monitors}
        except Exception as e:
            print(f"Warning: Could not list existing monitors: {e}")
            existing_names = {}

        for monitor_def in monitors_def.get("monitors", []):
            try:
                name = monitor_def["name"]

                if name in existing_names:
                    print(f"Updating monitor: {name}")
                    response = api.update_monitor(existing_names[name], monitor_def)
                    print(f"  Updated: ID {response.id}")
                else:
                    print(f"Creating monitor: {name}")
                    response = api.create_monitor(monitor_def)
                    print(f"  Created: ID {response.id}")

                deployed.append(response.id)

            except Exception as e:
                print(f"Error deploying monitor '{monitor_def.get('name', 'unknown')}': {e}")

    return deployed


def main():
    print("=" * 60)
    print("AI Safety Evals - Datadog Deployment")
    print("=" * 60)
    print()

    config = get_config()

    # Deploy dashboard
    print("Deploying Dashboard...")
    print("-" * 40)
    dashboard_id = deploy_dashboard(config)
    print()

    # Deploy monitors
    print("Deploying Monitors...")
    print("-" * 40)
    monitor_ids = deploy_monitors(config)
    print()

    # Summary
    print("=" * 60)
    print("Deployment Summary")
    print("=" * 60)
    if dashboard_id:
        site = os.getenv("DD_SITE", "ap2.datadoghq.com")
        print(f"Dashboard: https://{site}/dashboard/{dashboard_id}")
    print(f"Monitors deployed: {len(monitor_ids)}")
    print()
    print("View in Datadog:")
    print(f"  - Dashboards: https://{os.getenv('DD_SITE', 'ap2.datadoghq.com')}/dashboard/lists")
    print(f"  - Monitors: https://{os.getenv('DD_SITE', 'ap2.datadoghq.com')}/monitors/manage")
    print(f"  - LLM Observability: https://{os.getenv('DD_SITE', 'ap2.datadoghq.com')}/llm")


if __name__ == "__main__":
    main()
