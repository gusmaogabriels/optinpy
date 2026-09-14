"""Execute the main README's actual snippets so CI protects the usage guide."""
from copy import deepcopy
from pathlib import Path
import re

import numpy as np

import optinpy


def test_main_readme_python_examples():
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    source = readme.read_text(encoding="utf-8")
    blocks = list(re.finditer(r'^```python\n(.*?)^```\s*$', source, re.M | re.S))
    assert blocks, 'The main README must retain runnable usage examples'
    namespace = {'__name__': '__readme__'}
    saved_params = deepcopy(optinpy.params)
    try:
        for block in blocks:
            line = source[:block.start(1)].count('\n') + 1
            exec(compile('\n'*(line-1) + block.group(1), str(readme), 'exec'), namespace)
            if 'result' in namespace:
                assert namespace['result']['success']
        np.testing.assert_allclose(namespace['n3'].solution['f'], 58., atol=1e-7)
        np.testing.assert_allclose(namespace['solution']['x'], [2., 2.], atol=1e-7)
        np.testing.assert_allclose(namespace['result']['x'][-1], [1., 1.], atol=2e-5)
        np.testing.assert_allclose(namespace['constrained']['x'], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(namespace['solutions']['x'], [[1., 2.], [-2., 1.]], atol=1e-6)
    finally:
        optinpy.params.clear()
        optinpy.params.update(saved_params)
