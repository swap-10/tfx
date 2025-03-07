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
"""Async pipeline for testing."""

from tfx.dsl.compiler import compiler
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.decorators import component
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts


@component
def my_example_gen(examples: OutputArtifact[standard_artifacts.Examples]):
  del examples


@component
def my_transform(
    examples: InputArtifact[standard_artifacts.Examples],
    transform_graph: OutputArtifact[standard_artifacts.TransformGraph]):
  del examples, transform_graph


@component
def my_trainer(
    examples: InputArtifact[standard_artifacts.Examples],
    transform_graph: InputArtifact[standard_artifacts.TransformGraph],
    model: OutputArtifact[standard_artifacts.Model]):
  del examples, transform_graph, model


def create_pipeline() -> pipeline_pb2.Pipeline:
  """Creates an async pipeline for testing."""
  # pylint: disable=no-value-for-parameter
  example_gen = my_example_gen()
  transform = my_transform(examples=example_gen.outputs['examples'])
  trainer = my_trainer(
      examples=example_gen.outputs['examples'],
      transform_graph=transform.outputs['transform_graph'])
  # pylint: enable=no-value-for-parameter

  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          example_gen,
          transform,
          trainer,
      ],
      execution_mode=pipeline_lib.ExecutionMode.ASYNC)
  dsl_compiler = compiler.Compiler()
  compiled_pipeline: pipeline_pb2.Pipeline = dsl_compiler.compile(pipeline)

  # Compiler does not support setting min_count yet, so we mutate the proto
  # explicitly for testing.
  trainer = compiled_pipeline.nodes[2].pipeline_node
  assert trainer.node_info.id == 'my_trainer'
  for value in trainer.inputs.inputs.values():
    value.min_count = 1

  return compiled_pipeline
