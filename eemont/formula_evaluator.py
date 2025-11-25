"""
Formula evaluator for spectral indices.
Safely evaluates formulas from the awesome spectral indices list as Earth Engine operations.
"""
import re
import ee
from typing import Dict, Any, Optional


class FormulaEvaluator:
    """Safely evaluates spectral index formulas as Earth Engine operations."""
    
    def __init__(self, image: ee.Image, band_vars: Dict[str, ee.Image], params: Dict[str, float]):
        """
        Initialize formula evaluator.
        
        Parameters
        ----------
        image : ee.Image
            Source image
        band_vars : dict
            Dictionary mapping band abbreviations (N, R, G, B, etc.) to ee.Image bands
        params : dict
            Dictionary of parameters (G, C1, C2, L, etc.)
        """
        self.image = image
        self.band_vars = band_vars
        self.params = params
    
    def evaluate(self, formula_str: str) -> ee.Image:
        """
        Evaluate a formula string as Earth Engine operations.
        
        Parameters
        ----------
        formula_str : str
            Formula string from awesome list (e.g., "(N - R)/(N + R)")
        
        Returns
        -------
        ee.Image
            Computed index as single-band image
        """
        # Replace parameters first (to avoid conflicts with band names)
        formula = formula_str
        for param_name, param_value in self.params.items():
            # Replace parameter references with their values
            pattern = r'\b' + re.escape(param_name) + r'\b'
            formula = re.sub(pattern, str(param_value), formula)
        
        # Replace band abbreviations with actual band variables
        # Sort by length (longest first) to avoid partial matches
        sorted_bands = sorted(self.band_vars.keys(), key=len, reverse=True)
        for abbrev in sorted_bands:
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            # Replace with a placeholder that we'll handle
            formula = re.sub(pattern, f'__BAND_{abbrev}__', formula)
        
        # Now build the expression by replacing placeholders with actual operations
        # This is a simplified parser - for complex formulas, we may need a full AST parser
        # For now, handle common patterns
        
        # Replace placeholders back with band variables
        for abbrev, band_img in self.band_vars.items():
            formula = formula.replace(f'__BAND_{abbrev}__', f'band_vars["{abbrev}"]')
        
        # Use a safe evaluation approach
        # Create a namespace with only what we need
        namespace = {
            'band_vars': self.band_vars,
            'ee': ee,
        }
        
        # Add math operations that work with EE
        def safe_operations():
            """Create safe operation wrappers for EE"""
            return {
                'sqrt': lambda x: x.sqrt() if isinstance(x, ee.Image) else ee.Image.constant(x).sqrt(),
                'pow': lambda x, y: x.pow(y) if isinstance(x, ee.Image) else ee.Image.constant(x).pow(y),
                'exp': lambda x: x.exp() if isinstance(x, ee.Image) else ee.Image.constant(x).exp(),
                'log': lambda x: x.log() if isinstance(x, ee.Image) else ee.Image.constant(x).log(),
            }
        
        namespace.update(safe_operations())
        
        # For safety, we'll parse and build the expression manually for common cases
        # This is a simplified approach - for production, consider a proper formula parser
        
        # Try to evaluate using a restricted eval
        try:
            # Build expression safely
            result = self._build_expression(formula_str)
            return result
        except Exception as e:
            # Fallback: try eval with restricted namespace (less safe but may work)
            try:
                result = eval(formula, {"__builtins__": {}}, namespace)
                if isinstance(result, ee.Image):
                    return result
                else:
                    raise ValueError(f"Formula evaluation did not return an ee.Image: {type(result)}")
            except Exception as e2:
                raise ValueError(f"Failed to evaluate formula '{formula_str}': {e2}")
    
    def _build_expression(self, formula_str: str) -> ee.Image:
        """
        Build Earth Engine expression from formula string.
        This is a simplified parser that handles common patterns.
        """
        # For now, use a simple approach: replace and evaluate
        # This works for simple formulas but may fail for complex ones
        
        # Replace parameters
        formula = formula_str
        for param, value in self.params.items():
            formula = re.sub(r'\b' + re.escape(param) + r'\b', str(value), formula)
        
        # Replace band abbreviations with band variables (using placeholders)
        sorted_bands = sorted(self.band_vars.keys(), key=len, reverse=True)
        replacements = {}
        for i, abbrev in enumerate(sorted_bands):
            placeholder = f'__B{i}__'
            replacements[placeholder] = self.band_vars[abbrev]
            formula = re.sub(r'\b' + re.escape(abbrev) + r'\b', placeholder, formula)
        
        # Now we need to parse the formula and build EE operations
        # This is complex - for now, use a simpler approach with eval but restricted
        
        # Create safe namespace
        safe_dict = {'band_vars': self.band_vars, 'ee': ee}
        
        # Replace placeholders in formula with band_vars references
        for placeholder, band_img in replacements.items():
            # Find which abbrev this corresponds to
            for abbrev, img in self.band_vars.items():
                if img is band_img:
                    formula = formula.replace(placeholder, f'band_vars["{abbrev}"]')
                    break
        
        # Evaluate (with restrictions)
        result = eval(formula, {"__builtins__": {}}, safe_dict)
        
        if not isinstance(result, ee.Image):
            raise ValueError(f"Formula did not produce an ee.Image: {type(result)}")
        
        return result



