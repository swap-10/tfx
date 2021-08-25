# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO(kennethyang): DO NOT SUBMIT without one-line documentation for manual_node.

TODO(kennethyang): DO NOT SUBMIT without a detailed description of manual_node.
"""

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.types import component_spec

# Constant to access a manual node's description property.
MANUAL_NODE_DESCRIPTION_KEY = 'description'


class ManualNodeSpec(types.ComponentSpec):

  PARAMETERS = {
      MANUAL_NODE_DESCRIPTION_KEY: component_spec.ExecutionParameter(type=str)
  }
  INPUTS = {}
  OUTPUTS = {}


class ManualNode(base_component.BaseComponent):

  SPEC_CLASS = ManualNodeSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, description: str):
    spec = ManualNodeSpec(description=description)
    super().__init__(spec=spec)
