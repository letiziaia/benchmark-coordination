import pytest
import pandas as pd
from benchmark_coordination.pipeline.pipeline import Pipeline
from benchmark_coordination.network_builder.similarity_net import (
    build_similarity_network,
)
from benchmark_coordination.network_builder.thresholding import filter_edgelist


@pytest.fixture
def sample_data():
    data = pd.DataFrame(
        {
            "author_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "trace": ["a", "b", "c", "b", "c", "d", "e", "f", "d", "e"],
            "timestamp": [
                "2024-01-01 08:00:00",
                "2024-01-01 08:01:00",
                "2024-01-01 08:02:00",
                "2024-01-01 08:03:00",
                "2024-01-01 08:04:00",
                "2024-01-01 08:05:00",
                "2024-01-01 08:06:00",
                "2024-01-01 08:07:00",
                "2024-01-01 08:08:00",
                "2024-01-01 08:09:00",
            ],
        }
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data


def test_pipeline_fit(sample_data):
    """
    Test the .fit() method of the Pipeline class.
    """
    pipe = Pipeline(
        steps=[
            (
                "build_similarity_network",
                build_similarity_network,
                {"score": "jaccard", "symmetric": True},
            ),
            (
                "filter_edgelist",
                filter_edgelist,
                {"column_name": "similarity", "threshold": 0.3, "comparison": ">="},
            ),
        ],
        verbose=True,
    )
    result = pipe.fit(sample_data)
    expected_columns = ["source", "target", "similarity"]
    assert all(
        column in result.columns for column in expected_columns
    ), f"Expected columns {expected_columns} but got {result.columns}"
    assert len(result) == 4, f"Expected 4 rows but got {len(result)}"


def test_pipeline_length():
    """
    Test the __len__ method of the Pipeline class.
    """
    pipe = Pipeline(
        steps=[
            (
                "build_similarity_network",
                build_similarity_network,
                {"score": "jaccard", "symmetric": True},
            ),
            (
                "filter_edgelist",
                filter_edgelist,
                {"column_name": "similarity", "threshold": 0.3, "comparison": ">="},
            ),
        ]
    )
    assert len(pipe) == 2, f"A Pipeline with 2 steps should have len 2, not {len(pipe)}"


def test_pipeline_named_steps():
    """
    Test the named_steps property of the Pipeline class.
    """
    pipe = Pipeline(
        steps=[
            (
                "build_similarity_network",
                build_similarity_network,
                {"score": "jaccard", "symmetric": True},
            ),
            (
                "filter_edgelist",
                filter_edgelist,
                {"column_name": "similarity", "threshold": 0.3, "comparison": ">="},
            ),
        ]
    )
    named_steps = pipe.named_steps
    assert (
        "build_similarity_network" in named_steps
    ), "Missing step 'build_similarity_network'"
    assert "filter_edgelist" in named_steps, "Missing step 'filter_edgelist'"
    assert callable(named_steps["build_similarity_network"]), "Step is not callable"
    assert callable(named_steps["filter_edgelist"]), "Step is not callable"


def test_pipeline_named_steps_empty():
    """
    Test the named_steps property of the Pipeline class when the Pipeline has no steps.
    """
    pipe = Pipeline(steps=[])
    named_steps = pipe.named_steps
    assert named_steps == {}, "Named steps should be an empty dictionary"


def test_pipeline_steps():
    """
    Test the steps property of the Pipeline class.
    """
    steps = [
        (
            "build_similarity_network",
            build_similarity_network,
            {"score": "jaccard", "symmetric": True},
        ),
        (
            "filter_edgelist",
            filter_edgelist,
            {"column_name": "similarity", "threshold": 0.3, "comparison": ">="},
        ),
    ]
    pipe = Pipeline(steps=steps)
    assert pipe.steps == steps, "Steps property should return the steps"


def test_pipeline_repr():
    """
    Test the __repr__ method of the Pipeline class.
    """
    pipe = Pipeline(
        steps=[
            (
                "build_similarity_network",
                build_similarity_network,
                {"score": "jaccard", "symmetric": True},
            ),
            (
                "filter_edgelist",
                filter_edgelist,
                {"column_name": "similarity", "threshold": 0.3, "comparison": ">="},
            ),
        ]
    )
    assert repr(pipe).startswith(
        "Pipeline(steps=[('build_similarity_network', <function build_similarity_network at 0x"
    ), f"Unexpected repr: {repr(pipe)}"
