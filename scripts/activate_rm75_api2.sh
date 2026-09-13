#!/usr/bin/env bash
# Activate the workspace-local, ROS-free RM75 API2 Python dependencies.
_RM75_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_RM75_PYTHON="${_RM75_ROOT}/.rm75_python"
_RM75_SITE="${_RM75_PYTHON}/cmeel.prefix/lib/python$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')/site-packages"
export PYTHONPATH="${_RM75_PYTHON}:${_RM75_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${_RM75_PYTHON}/cmeel.prefix/lib:${_RM75_PYTHON}/cmeel.prefix/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [ -x "${_RM75_ROOT}/.tools/platform-tools/adb" ]; then
  export PATH="${_RM75_ROOT}/.tools/platform-tools:${PATH}"
fi
unset _RM75_ROOT _RM75_PYTHON _RM75_SITE
echo "RM75 API2 environment active (ROS-free)"
