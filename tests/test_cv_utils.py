"""
Tests for utils/cv_utils.py.

CV result helpers used by the GS-CV / SBI-CV plotting pipeline.
"""


class TestParamsToStr:
    def test_returns_string(self):
        from utils.cv_utils import params_to_str
        params = {'sigma_percep': 0.1, 'eta_learning': 0.3}
        result = params_to_str(params)
        assert isinstance(result, str)
        assert len(result) > 0
