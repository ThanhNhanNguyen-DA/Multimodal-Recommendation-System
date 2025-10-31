import pytest
from src.recommender import Recommender

@pytest.fixture
def recommender():
    return Recommender()

def test_recommend_by_text(recommender):
    text_input = "Stylish summer dress"
    recommendations = recommender.recommend_by_text(text_input, top_k=5)
    assert len(recommendations) == 5
    assert all(isinstance(rec, dict) for rec in recommendations)

def test_recommend_by_image(recommender):
    image_path = "path/to/test/image.jpg"
    recommendations = recommender.recommend_by_image(image_path, top_k=5)
    assert len(recommendations) == 5
    assert all(isinstance(rec, dict) for rec in recommendations)

def test_fusion_method(recommender):
    text_input = "Casual t-shirt"
    image_path = "path/to/test/image.jpg"
    recommendations = recommender.recommend_by_fusion(text_input, image_path, top_k=5)
    assert len(recommendations) == 5
    assert all(isinstance(rec, dict) for rec in recommendations)