import pytest
import numpy as np
from app.models.request import FaceDetection, Landmarks, Cluster
from app.models.request import ClusterPayload
from app.pipeline.filters import drop_noise_clusters, pre_filter_face, pre_filter_faces
from app.pipeline.dedup import temporal_dedup
from app.pipeline.scoring import score_frontality, score_size, score_illumination, score_composite, score_all, select_best, generate_identity
from tests.conftest import make_face, make_cluster, make_payload


class TestNoiseFilter:
    def test_noise_cluster_dropped(self):
        c = make_cluster(cluster_id=-1)
        result = drop_noise_clusters([c])
        assert len(result) == 0

    def test_valid_cluster_kept(self):
        c = make_cluster(cluster_id=0)
        result = drop_noise_clusters([c])
        assert len(result) == 1

    def test_mixed_clusters(self):
        clusters = [make_cluster(0), make_cluster(-1), make_cluster(1), make_cluster(-1), make_cluster(2)]
        result = drop_noise_clusters(clusters)
        assert len(result) == 3
        assert all(c.id != -1 for c in result)


class TestPreFilter:
    def test_passes_all_thresholds(self):
        f = make_face(confidence=0.9, quality_score=0.8, bbox=[100, 80, 200, 180])
        assert pre_filter_face(f) is True

    def test_fails_tiny_face(self):
        f = make_face(bbox=[0, 0, 30, 30])
        assert pre_filter_face(f) is False

    def test_fails_low_confidence(self):
        f = make_face(confidence=0.50)
        assert pre_filter_face(f) is False

    def test_fails_low_quality(self):
        f = make_face(quality_score=0.10)
        assert pre_filter_face(f) is False

    def test_fails_side_face(self):
        f = make_face(
            left_eye=[100, 100],
            right_eye=[200, 150],
            nose=[100, 130],
            bbox=[50, 50, 250, 250],
        )
        assert pre_filter_face(f) is False

    def test_fallback_all_fail(self):
        faces = [
            make_face(confidence=0.1),
            make_face(confidence=0.2),
        ]
        result = pre_filter_faces(faces)
        assert len(result) == len(faces)


class TestDeduplicator:
    def test_dedup_keeps_distant_frames(self):
        faces = [make_face(timestamp_sec=0.0), make_face(timestamp_sec=2.0), make_face(timestamp_sec=4.0)]
        scored = score_all(faces)
        result = temporal_dedup(scored)
        assert len(result) == 3

    def test_dedup_collapses_same_window(self):
        faces = [make_face(timestamp_sec=0.0), make_face(timestamp_sec=0.3), make_face(timestamp_sec=0.7)]
        scored = score_all(faces)
        result = temporal_dedup(scored)
        assert len(result) == 1

    def test_dedup_selects_better_score(self):
        f1 = make_face(timestamp_sec=0.2, quality_score=0.5)
        f2 = make_face(timestamp_sec=0.5, quality_score=0.9)
        scored = score_all([f1, f2])
        result = temporal_dedup(scored)
        assert len(result) == 1
        assert result[0].quality_score == 0.9

    def test_dedup_single_face(self):
        faces = [make_face(timestamp_sec=0.0)]
        scored = score_all(faces)
        result = temporal_dedup(scored)
        assert len(result) == 1

    def test_dedup_boundary_exact(self):
        faces = [make_face(timestamp_sec=0.0), make_face(timestamp_sec=1.0)]
        scored = score_all(faces)
        result = temporal_dedup(scored)
        assert len(result) == 2


class TestScorer:
    def test_frontality_perfect(self):
        f = make_face(left_eye=[130, 110], right_eye=[170, 110], nose=[150, 110])
        score = score_frontality(f)
        assert abs(score - 1.0) < 0.01

    def test_frontality_side_face(self):
        f = make_face(left_eye=[100, 100], right_eye=[200, 150], nose=[100, 130])
        score = score_frontality(f)
        assert score < 0.4

    def test_frontality_no_eye_span(self):
        f = make_face(left_eye=[150, 110], right_eye=[150, 110], nose=[150, 110])
        score = score_frontality(f)
        assert score >= 0.0

    def test_size_large_face(self):
        f = make_face(bbox=[0, 0, 300, 300])
        score = score_size(f)
        assert score == 1.0

    def test_size_tiny_face(self):
        f = make_face(bbox=[0, 0, 30, 30])
        score = score_size(f)
        assert score < 0.02

    def test_composite_weights_sum(self):
        f = make_face(quality_score=1.0, confidence=1.0, bbox=[0, 0, 250, 250])
        score = score_composite(f)
        assert 0.0 <= score <= 1.0

    def test_composite_deterministic(self):
        f = make_face()
        s1 = score_composite(f)
        s2 = score_composite(f)
        assert s1 == s2

    def test_score_all_attaches_score(self):
        faces = [make_face(), make_face()]
        scored = score_all(faces)
        for f in scored:
            assert hasattr(f, "_score")
            assert isinstance(f._score, float)


class TestSelector:
    def test_select_best_returns_highest_score(self):
        f1 = make_face(quality_score=0.3)
        f2 = make_face(quality_score=0.9)
        scored = score_all([f1, f2])
        best = select_best(scored)
        assert best is f2

    def test_select_best_empty(self):
        assert select_best([]) is None

    def test_select_best_tie_break_earlier_frame(self):
        f1 = make_face(frame_idx=10, quality_score=0.8)
        f2 = make_face(frame_idx=5, quality_score=0.8)
        scored = score_all([f1, f2])
        best = select_best(scored)
        assert best.frame_idx == 5


class TestIdentity:
    def test_identity_padding_4digits(self):
        person_id, uid = generate_identity(1, 1, 100)
        assert person_id == "person_0001"
        assert len(uid) == 36

    def test_identity_padding_5digits(self):
        person_id, uid = generate_identity(1, 1, 10000)
        assert person_id == "person_00001"

    def test_uuid_is_unique(self):
        _, uid1 = generate_identity(1, 1, 10)
        _, uid2 = generate_identity(1, 2, 10)
        assert uid1 != uid2
