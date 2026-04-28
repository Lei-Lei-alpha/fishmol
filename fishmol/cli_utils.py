import numpy as np
from typing import List, Union, Any

def parse_selection(selection_str: str, atoms: Any) -> List[int]:
    """
    Parse a CLI selection string into a list of atom indices.
    
    Supported formats:
    - Symbol: "O", "H" (returns all atoms of that element)
    - CSV Indices: "0,1,2,5"
    - Ranges: "0-10" (inclusive)
    - Mixed: "O,10-20,30"
    """
    if not selection_str:
        return []
    
    indices = []
    parts = selection_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Range: 0-10
        if '-' in part and not part.startswith('-'):
            try:
                start_str, end_str = part.split('-')
                start, end = int(start_str), int(end_str)
                indices.extend(list(range(start, end + 1)))
            except ValueError:
                pass
        # Numeric Index
        elif part.isdigit():
            indices.append(int(part))
        # Chemical Symbol
        else:
            # Case insensitive check
            symbs = np.char.upper(atoms.symbs)
            target = part.upper()
            match_idx = np.where(symbs == target)[0].tolist()
            if match_idx:
                indices.extend(match_idx)
            else:
                # Fallback: maybe it's a negative index?
                try:
                    indices.append(int(part))
                except ValueError:
                    print(f"Warning: Selection '{part}' not recognized as index or symbol.")
                    
    return sorted(list(set(indices)))

def parse_slice(slice_str: str) -> Union[slice, int, str]:
    """Parse a python-style slice string 'start:stop:step' or a single int."""
    if not slice_str or slice_str == ":":
        return ":"
    
    if ":" not in slice_str:
        try:
            return int(slice_str)
        except ValueError:
            return slice_str
            
    parts = slice_str.split(':')
    # Convert empty strings to None for slice()
    p = [int(x.strip()) if x.strip() else None for x in parts]
    
    if len(p) == 1:
        return slice(p[0], p[0] + 1)
    elif len(p) == 2:
        return slice(p[0], p[1])
    else:
        return slice(p[0], p[1], p[2])
