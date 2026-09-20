#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patch the msopgen-generated CMakePresets.json for the AFD op build.

Updates CMAKE_BUILD_TYPE, ENABLE_SOURCE_PACKAGE, and vendor_name (AFD uses
"afd-plugin" so the run package installs under vendors/afd-plugin).
"""

import argparse
import json
import sys


def update_json_path(args):
    """Update configuration items in the CMakePresets.json file."""
    try:
        with open(args.file_path) as f:
            data = json.load(f)

        # Iterate over the first configure preset (the one msopgen emits).
        configure_preset = data.get("configurePresets", [{}])[0]
        cache_variables = configure_preset.get("cacheVariables", {})

        if "CMAKE_BUILD_TYPE" in cache_variables:
            cache_variables["CMAKE_BUILD_TYPE"]["value"] = args.build_type
        else:
            print("CMAKE_BUILD_TYPE field not found")
            sys.exit(1)

        if "ENABLE_SOURCE_PACKAGE" in cache_variables:
            cache_variables["ENABLE_SOURCE_PACKAGE"]["value"] = args.enable_source
        else:
            print("ENABLE_SOURCE_PACKAGE field not found")
            sys.exit(1)

        if "vendor_name" in cache_variables:
            cache_variables["vendor_name"]["value"] = args.vendor_name
        else:
            print("vendor_name field not found")
            sys.exit(1)

        with open(args.file_path, "w") as f:
            json.dump(data, f, indent=4)
        print("Successfully updated parameters")

    except FileNotFoundError:
        print(f"File not found: {args.file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"JSON format error: {args.file_path}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify configuration items in CMakePresets.json"
    )
    parser.add_argument("file_path", help="Path to the JSON file")
    parser.add_argument("build_type", help="Build type (e.g., Debug or Release)")
    parser.add_argument(
        "enable_source", help="Enable source package generation (true/false)"
    )
    parser.add_argument(
        "vendor_name", help="Specify the custom operator directory name"
    )

    args = parser.parse_args()

    update_json_path(args)
