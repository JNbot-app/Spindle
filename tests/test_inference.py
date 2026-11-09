import pytest

from spindle import Spindle

TOOLS = [{'name': 'getWeather', 'description': 'Get the current weather for a city.',
          'parameters': {'type': 'object', 'properties': {'cityName': {'type': 'string'}},
                         'required': ['cityName']}}]


def test_tool_call_with_original_names(model_path):
    try:
        model = Spindle(model_path, TOOLS)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    with model:
        result = model('What is the weather in Lagos?')
    assert result['complete']
    assert result['function_calls'] == [{'name': 'getWeather', 'arguments': {'cityName': 'Lagos'}}]
    assert 0 < result['call_probabilities'][0] <= 1
