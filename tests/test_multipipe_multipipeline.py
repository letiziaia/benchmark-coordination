import pytest
import pandas as pd
from benchmark_coordination.pipeline.pipeline import Pipeline
from benchmark_coordination.pipeline.multipipe import MultiPipeline
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


@pytest.fixture
def sample_pipeline():
    return Pipeline(
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


def test_multipipeline_fit(sample_data, sample_pipeline):
    """
    Test the .fit() method of the MultiPipeline class.
    """
    pipe = MultiPipeline(pipelines=[sample_pipeline, sample_pipeline])
    result = pipe.fit(sample_data)
    expected_columns = ["source", "target", "similarity"]

    assert isinstance(result, list), f"Expected a list but got {type(result)}"
    assert len(result) == 2, f"Expected 2 results but got {len(result)}"
    for res in result:
        assert isinstance(
            res, pd.DataFrame
        ), f"Expected a DataFrame but got {type(res)}"
        assert all(
            column in res.columns for column in expected_columns
        ), f"Expected columns {expected_columns} but got {res.columns}"


def test_multipipeline_length():
    """
    Test the __len__ method of the MultiPipeline class.
    """
    pipe = MultiPipeline(
        pipelines=[
            Pipeline(
                steps=[("a", lambda x: x)],
            ),
            Pipeline(
                steps=[("a", lambda x: x)],
            ),
        ]
    )
    assert len(pipe) == 2, f"Expected 2 but got {len(pipe)}"


def test_multipipeline_named_pipelines():
    """
    Test the named_pipelines property of the MultiPipeline class.
    """
    pipe = MultiPipeline(
        pipelines=[
            Pipeline(steps=[("a", lambda x: x)], pipeline_id="a"),
            Pipeline(steps=[("a", lambda x: x)], pipeline_id="b"),
        ]
    )
    assert pipe.named_pipelines == [
        "a",
        "b",
    ], f"Expected ['a', 'b'] but got {pipe.named_pipelines}"

    pipe = MultiPipeline(pipelines=[])
    assert pipe.named_pipelines == [], f"Expected [] but got {pipe.named_pipelines}"


def test_multipipeline_pipeline_ids():
    """
    Test the _pipeline_ids method of the MultiPipeline class.
    """
    pipe = MultiPipeline(
        pipelines=[
            Pipeline(steps=[("a", lambda x: x)], pipeline_id="a"),
            Pipeline(steps=[("a", lambda x: x)], pipeline_id="b"),
        ]
    )
    assert pipe._pipeline_ids() == [
        "a",
        "b",
    ], f"Expected ['a', 'b'] but got {pipe._pipeline_ids()}"

    pipe = MultiPipeline(pipelines=[])
    assert pipe._pipeline_ids() == [], f"Expected [] but got {pipe._pipeline_ids()}"


def test_multipipeline_fit_pipeline(sample_data, sample_pipeline):
    """
    Test the _fit_pipeline method of the MultiPipeline class.
    """
    pipe = MultiPipeline(pipelines=[sample_pipeline, sample_pipeline])
    result = pipe._fit_pipeline(sample_pipeline, sample_data)
    expected_columns = ["source", "target", "similarity"]

    assert isinstance(result, tuple), f"Expected a tuple but got {type(result)}"
    assert len(result) == 2, f"Expected a tuple of length 2 but got {len(result)}"
    assert isinstance(result[0], str), f"Expected a string but got {type(result[0])}"
    assert isinstance(
        result[1], pd.DataFrame
    ), f"Expected a DataFrame but got {type(result[1])}"
    assert all(
        column in result[1].columns for column in expected_columns
    ), f"Expected columns {expected_columns} but got {result[1].columns}"


def test_multipipeline_get_item(sample_pipeline):
    """
    Test the __getitem__ method of the MultiPipeline class.
    """
    pipe = MultiPipeline(pipelines=[sample_pipeline, sample_pipeline])
    assert pipe[0] == sample_pipeline, f"Expected {sample_pipeline} but got {pipe[0]}"
    assert pipe[1] == sample_pipeline, f"Expected {sample_pipeline} but got {pipe[1]}"

    with pytest.raises(IndexError):
        pipe[2]
