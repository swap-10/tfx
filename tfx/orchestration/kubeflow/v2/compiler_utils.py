# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Utility methods for Kubeflow V2 pipeline compilation."""
# TODO(b/172080784): Add more tests for this module.

import itertools
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
from tfx import types
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import parameter_utils
from tfx.types import artifact
from tfx.types import channel
from tfx.types import standard_artifacts
from tfx.types.experimental import simple_artifacts
from tfx.utils import json_utils
import yaml

from google.protobuf import struct_pb2
from google.protobuf import json_format
from google.protobuf import message
from ml_metadata.proto import metadata_store_pb2

# Key of TFX type path and name in artifact custom properties.
TFX_TYPE_KEY = 'tfx_type'
TYPE_NAME_KEY = 'type_name'

_SUPPORTED_STANDARD_ARTIFACT_TYPES = frozenset(
    (standard_artifacts.ExampleAnomalies, standard_artifacts.ExampleStatistics,
     standard_artifacts.Examples, standard_artifacts.HyperParameters,
     standard_artifacts.InferenceResult, standard_artifacts.InfraBlessing,
     standard_artifacts.Model, standard_artifacts.ModelBlessing,
     standard_artifacts.ModelEvaluation, standard_artifacts.ModelRun,
     standard_artifacts.PushedModel, standard_artifacts.Schema,
     standard_artifacts.TransformGraph, standard_artifacts.TransformCache,
     standard_artifacts.Float, standard_artifacts.Integer,
     standard_artifacts.String, simple_artifacts.Metrics,
     simple_artifacts.Statistics, simple_artifacts.Dataset,
     simple_artifacts.File))


def _get_full_class_path(klass: Type[types.Artifact]) -> str:
  return klass.__module__ + '.' + klass.__qualname__


TITLE_TO_CLASS_PATH = {
    f'tfx.{klass.__qualname__}': _get_full_class_path(klass)
    for klass in _SUPPORTED_STANDARD_ARTIFACT_TYPES
}

# Keywords used in artifact type YAML specs.
_YAML_INT_TYPE = 'int'
_YAML_STRING_TYPE = 'string'
_YAML_DOUBLE_TYPE = 'double'


def build_runtime_parameter_spec(
    parameters: List[data_types.RuntimeParameter]
) -> Dict[str, pipeline_pb2.PipelineSpec.RuntimeParameter]:
  """Converts RuntimeParameters to mapping from names to proto messages."""

  def to_message(parameter: data_types.RuntimeParameter):
    """Converts a RuntimeParameter to RuntimeParameter message."""
    result = pipeline_pb2.PipelineSpec.RuntimeParameter()
    # 1. Map the RuntimeParameter type to an enum in the proto definition.
    if parameter.ptype == int or parameter.ptype == bool:
      result.type = pipeline_pb2.PrimitiveType.INT
    elif parameter.ptype == float:
      result.type = pipeline_pb2.PrimitiveType.DOUBLE
    elif parameter.ptype == str:
      result.type = pipeline_pb2.PrimitiveType.STRING
    else:
      raise TypeError(
          'Unknown parameter type: {} found in parameter: {}'.format(
              parameter.ptype, parameter.name))
    # 2. Convert its default value.
    default = value_converter(parameter.default)
    if default is not None:
      result.default_value.CopyFrom(default.constant_value)
    return result

  return {param.name: to_message(param) for param in parameters}


def build_parameter_type_spec(
    value: Union[types.Property, data_types.RuntimeParameter]
) -> pipeline_pb2.ComponentInputsSpec.ParameterSpec:
  """Extracts the artifact type info into ComponentInputsSpec.ParameterSpec."""
  is_runtime_param = isinstance(value, data_types.RuntimeParameter)
  result = pipeline_pb2.ComponentInputsSpec.ParameterSpec()
  if isinstance(value, int) or (is_runtime_param and value.ptype == int):
    result.type = pipeline_pb2.PrimitiveType.PrimitiveTypeEnum.INT
  elif isinstance(value, float) or (is_runtime_param and value.ptype == float):
    result.type = pipeline_pb2.PrimitiveType.PrimitiveTypeEnum.DOUBLE
  elif isinstance(value, str) or (is_runtime_param and value.ptype == str):
    result.type = pipeline_pb2.PrimitiveType.PrimitiveTypeEnum.STRING
  else:
    # By default, unrecognized object will be json dumped, hence is string type.
    # For example, resolver class.
    result.type = pipeline_pb2.PrimitiveType.PrimitiveTypeEnum.STRING
  return result


def _validate_properties_schema(
    instance_schema: str,
    properties: Optional[Mapping[str, artifact.PropertyType]] = None):
  """Validates the declared property types are consistent with the schema.

  Args:
    instance_schema: YAML string of the artifact property schema.
    properties: The actual property schema of an Artifact Python class.

  Raises:
    KeyError: When actual property have additional properties than what's
      specified in the YAML schema.
    TypeError: When the same property is declared with different types in YAML
      schema and the Artifact Python class.
  """
  schema = yaml.safe_load(instance_schema).get('properties', {})
  properties = properties or {}

  for k, v in properties.items():
    if k not in schema:
      raise KeyError('Actual property: {} not expected in artifact type schema:'
                     ' {}'.format(k, schema))
    # It's okay that we only validate the constant_value case, since
    # RuntimeParameter's ptype should be validated during component
    # instantiation.
    # We only validate primitive-typed property for now because other types can
    # have nested schema in the YAML spec as well.
    if (schema[k]['type'] == _YAML_INT_TYPE and
        v.type != artifact.PropertyType.INT or
        schema[k]['type'] == _YAML_STRING_TYPE and
        v.type != artifact.PropertyType.STRING or
        schema[k]['type'] == _YAML_DOUBLE_TYPE and
        v.type != artifact.PropertyType.FLOAT):
      raise TypeError(f'Property type mismatched at {k} for schema: {schema}. '
                      f'Expected {schema[k]["type"]} but got {v.type}')


def build_input_artifact_spec(
    channel_spec: channel.Channel
) -> pipeline_pb2.ComponentInputsSpec.ArtifactSpec:
  """Builds artifact type spec for an input channel."""
  artifact_instance = channel_spec.type()
  result = pipeline_pb2.ComponentInputsSpec.ArtifactSpec()
  result.artifact_type.CopyFrom(
      pipeline_pb2.ArtifactTypeSchema(
          instance_schema=get_artifact_schema(artifact_instance)))
  _validate_properties_schema(
      instance_schema=result.artifact_type.instance_schema,
      properties=channel_spec.type.PROPERTIES)
  return result


def build_output_artifact_spec(
    channel_spec: channel.Channel
) -> pipeline_pb2.ComponentOutputsSpec.ArtifactSpec:
  """Builds artifact type spec for an output channel."""
  # We use the first artifact instance if available from channel, otherwise
  # create one.
  artifacts = list(channel_spec.get())
  artifact_instance = artifacts[0] if artifacts else channel_spec.type()

  result = pipeline_pb2.ComponentOutputsSpec.ArtifactSpec()
  result.artifact_type.CopyFrom(
      pipeline_pb2.ArtifactTypeSchema(
          instance_schema=get_artifact_schema(artifact_instance)))

  _validate_properties_schema(
      instance_schema=result.artifact_type.instance_schema,
      properties=channel_spec.type.PROPERTIES)

  struct_proto = pack_artifact_properties(artifact_instance)
  if struct_proto:
    result.metadata.CopyFrom(struct_proto)
  return result


def pack_artifact_properties(artifact_instance: artifact.Artifact):
  """Packs artifact properties and custom properties into a Struct proto."""
  struct_proto = struct_pb2.Struct()
  metadata = {}
  for k, v in itertools.chain(
      artifact_instance.mlmd_artifact.properties.items(),
      artifact_instance.mlmd_artifact.custom_properties.items()):
    metadata[k] = getattr(v, v.WhichOneof('value'))

  # JSON does not differetiate between int and double, but MLMD value type does.
  # MLMD int is stored as double as a result.
  struct_proto.update(metadata)
  return struct_proto


def value_converter(
    tfx_value: Any) -> Optional[pipeline_pb2.ValueOrRuntimeParameter]:
  """Converts TFX/MLMD values into Kubeflow pipeline ValueOrRuntimeParameter."""
  if tfx_value is None:
    return None

  result = pipeline_pb2.ValueOrRuntimeParameter()
  if isinstance(tfx_value, (int, float, str)):
    result.constant_value.CopyFrom(get_kubeflow_value(tfx_value))
  elif isinstance(tfx_value, (Dict, List)):
    result.constant_value.CopyFrom(
        pipeline_pb2.Value(string_value=json.dumps(tfx_value)))
  elif isinstance(tfx_value, data_types.RuntimeParameter):
    # Attach the runtime parameter to the context.
    parameter_utils.attach_parameter(tfx_value)
    result.runtime_parameter = tfx_value.name
  elif isinstance(tfx_value, metadata_store_pb2.Value):
    if tfx_value.WhichOneof('value') == 'int_value':
      result.constant_value.CopyFrom(
          pipeline_pb2.Value(int_value=tfx_value.int_value))
    elif tfx_value.WhichOneof('value') == 'double_value':
      result.constant_value.CopyFrom(
          pipeline_pb2.Value(double_value=tfx_value.double_value))
    elif tfx_value.WhichOneof('value') == 'string_value':
      result.constant_value.CopyFrom(
          pipeline_pb2.Value(string_value=tfx_value.string_value))
  elif isinstance(tfx_value, message.Message):
    result.constant_value.CopyFrom(
        pipeline_pb2.Value(
            string_value=json_format.MessageToJson(
                message=tfx_value, sort_keys=True)))
  else:
    # By default will attempt to encode the object using json_utils.dumps.
    result.constant_value.CopyFrom(
        pipeline_pb2.Value(string_value=json_utils.dumps(tfx_value)))
  return result


def get_kubeflow_value(
    tfx_value: Union[int, float, str]) -> Optional[pipeline_pb2.Value]:
  """Converts TFX/MLMD values into Kubeflow pipeline Value proto message."""
  if tfx_value is None:
    return None

  result = pipeline_pb2.Value()
  if isinstance(tfx_value, int):
    result.int_value = tfx_value
  elif isinstance(tfx_value, float):
    result.double_value = tfx_value
  elif isinstance(tfx_value, str):
    result.string_value = tfx_value
  else:
    raise TypeError('Got unknown type of value: {}'.format(tfx_value))

  return result


def get_mlmd_value(
    kubeflow_value: pipeline_pb2.Value) -> metadata_store_pb2.Value:
  """Converts Kubeflow pipeline Value pb message to MLMD Value."""
  result = metadata_store_pb2.Value()
  if kubeflow_value.WhichOneof('value') == 'int_value':
    result.int_value = kubeflow_value.int_value
  elif kubeflow_value.WhichOneof('value') == 'double_value':
    result.double_value = kubeflow_value.double_value
  elif kubeflow_value.WhichOneof('value') == 'string_value':
    result.string_value = kubeflow_value.string_value
  else:
    raise TypeError('Get unknown type of value: {}'.format(kubeflow_value))

  return result


def get_artifact_schema(artifact_instance: artifact.Artifact) -> str:
  """Gets the YAML schema string associated with the artifact type."""
  if isinstance(artifact_instance, tuple(_SUPPORTED_STANDARD_ARTIFACT_TYPES)):
    # For supported first-party artifact types, get the built-in schema yaml per
    # its type name.
    schema_path = os.path.join(
        os.path.dirname(__file__), 'artifact_types',
        '{}.yaml'.format(artifact_instance.type_name))
    return fileio.open(schema_path, 'rb').read()
  else:
    # Otherwise, fall back to the generic `Artifact` type schema.
    # To recover the Python type object at runtime, the class import path will
    # be encoded as the schema title.

    # Read the generic artifact schema template.
    schema_path = os.path.join(
        os.path.dirname(__file__), 'artifact_types', 'Artifact.yaml')
    data = yaml.safe_load(fileio.open(schema_path, 'rb').read())
    # Encode class import path.
    data['title'] = '%s.%s' % (artifact_instance.__class__.__module__,
                               artifact_instance.__class__.__name__)
    return yaml.dump(data, sort_keys=False)


def get_artifact_title(artifact_type: Type[artifact.Artifact]) -> str:
  """Gets the schema title from the artifact python class."""
  if artifact_type in _SUPPORTED_STANDARD_ARTIFACT_TYPES:
    return 'tfx.{}'.format(artifact_type.__name__)
  return 'tfx.Artifact'
