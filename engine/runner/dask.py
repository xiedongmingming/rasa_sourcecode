from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text

import dask

from rasa.engine.exceptions import GraphRunError
from rasa.engine.graph import ExecutionContext, GraphNode, GraphNodeHook, GraphSchema
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage

logger = logging.getLogger(__name__)


class DaskGraphRunner(GraphRunner):
    """
    Dask implementation of a `GraphRunner`.
    """

    def __init__(
            self,
            graph_schema: GraphSchema,
            model_storage: ModelStorage,
            execution_context: ExecutionContext,
            hooks: Optional[List[GraphNodeHook]] = None,
    ) -> None:
        """
        Initializes a `DaskGraphRunner`.

        Args:
            graph_schema: The graph schema that will be run.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            execution_context: Information about the current graph run to be passed to
                each node.
            hooks: These are called before and after the execution of each node.
        """
        self._graph_schema = graph_schema

        self._instantiated_nodes: Dict[Text, GraphNode] = self._instantiate_nodes(
            graph_schema, model_storage, execution_context, hooks
        )

        self._execution_context: ExecutionContext = execution_context

    @classmethod
    def create(
            cls,
            graph_schema: GraphSchema,
            model_storage: ModelStorage,
            execution_context: ExecutionContext,
            hooks: Optional[List[GraphNodeHook]] = None,
    ) -> DaskGraphRunner:
        """
        Creates the runner (see parent class for full docstring).
        """
        return cls(graph_schema, model_storage, execution_context, hooks)

    @staticmethod
    def _instantiate_nodes(
            graph_schema: GraphSchema,
            model_storage: ModelStorage,
            execution_context: ExecutionContext,
            hooks: Optional[List[GraphNodeHook]] = None,
    ) -> Dict[Text, GraphNode]:
        return {
            node_name: GraphNode.from_schema_node(
                node_name, schema_node, model_storage, execution_context, hooks
            )
            for node_name, schema_node in graph_schema.nodes.items()
        }

    def _build_dask_graph(self, schema: GraphSchema) -> Dict[Text, Any]:
        """
        Builds a dask graph from the instantiated graph.

        For more information about dask graphs
        see: https://docs.dask.org/en/latest/spec.html
        """
        run_graph = {
            node_name: (
                self._instantiated_nodes[node_name],
                *schema_node.needs.values(),
            )
            for node_name, schema_node in schema.nodes.items()
        }
        return run_graph

    def run(
            self,
            inputs: Optional[Dict[Text, Any]] = None,  # {'__importer__': E2EImporter}
            targets: Optional[List[Text]] = None,  # None
    ) -> Dict[Text, Any]:
        """
        Runs the graph (see parent class for full docstring).
        """
        run_targets = targets if targets else self._graph_schema.target_names # ['schema_validator', 'finetuning_validator', 'nlu_training_data_provider', 'train_JiebaTokenizer0', 'run_JiebaTokenizer0', 'run_LanguageModelFeaturizer1', 'train_DIETClassifier2', 'train_ResponseSelector3', 'domain_provider', 'domain_for_core_training_provider', 'story_graph_provider', 'training_tracker_provider', 'train_MemoizationPolicy0', 'train_TEDPolicy1', 'train_RulePolicy2']

        minimal_schema = self._graph_schema.minimal_graph_schema(run_targets)

        run_graph = self._build_dask_graph(minimal_schema)

        if inputs:
            #
            self._add_inputs_to_graph(inputs, run_graph)

        logger.debug(
            f"Running graph with inputs: {inputs}, targets: {targets} "
            f"and {self._execution_context}."
        )

        try:

            dask_result = dask.get(run_graph, run_targets)
            # (
            #       ('train_JiebaTokenizer0', Resource(name='train_JiebaTokenizer0', output_fingerprint='e2a76e6612274a97ac0480c654f8113b')),
            #       ('train_DIETClassifier2', Resource(name='train_DIETClassifier2', output_fingerprint='1acc076b85ae4c178728d9d42cc03aab')),
            #       ('train_ResponseSelector3', Resource(name='train_ResponseSelector3', output_fingerprint='88bc89dd5e6c4f5694191470dfb18c43')),
            #       ('domain_provider', Domain: 32 actions, 8 intents, 17 responses, 1 slots, 0 entities, 0 forms),
            #       ('train_MemoizationPolicy0', Resource(name='train_MemoizationPolicy0', output_fingerprint='ede657794a564b009c9630da7e75dd9c')),
            #       ('train_TEDPolicy1', Resource(name='train_TEDPolicy1', output_fingerprint='6144e0183d0f40838ea619367ad348b0')),
            #       ('train_RulePolicy2', Resource(name='train_RulePolicy2', output_fingerprint='3b703e36b7be44c9ae8abd910f2921b6'))
            # )
            return dict(dask_result)

        except RuntimeError as e:

            raise GraphRunError("Error running runner.") from e

    @staticmethod
    def _add_inputs_to_graph(inputs: Optional[Dict[Text, Any]], graph: Any) -> None:

        if inputs is None:
            return

        for input_name, input_value in inputs.items():

            if isinstance(input_value, str) and input_value in graph.keys():
                #
                raise GraphRunError(
                    f"Input value '{input_value}' clashes with a node name. Make sure "
                    f"that none of the input names passed to the `run` method are the "
                    f"same as node names in the graph schema."
                )

            graph[input_name] = (input_name, input_value)
