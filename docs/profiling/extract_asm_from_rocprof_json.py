#!/usr/bin/env python3
import json
import sys

def extract_assembly_from_json(json_file_path, output_file_path):
    """
    Parse the JSON file and extract only the assembly code.
    The JSON structure appears to be: {"code": [["instruction", ...], ...]}
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the code array
        if 'code' not in data:
            print("Error: 'code' key not found in JSON")
            return False
        
        code_array = data['code']
        
        # Extract just the assembly instructions (first element of each array)
        assembly_lines = []
        for item in code_array:
            if isinstance(item, list) and len(item) > 0:
                instruction = item[0]
                if isinstance(instruction, str) and instruction.strip():
                    assembly_lines.append(instruction)
        
        # Write to output file
        with open(output_file_path, 'w') as f:
            for line in assembly_lines:
                f.write(line + '\n')
        
        print(f"Successfully extracted {len(assembly_lines)} assembly instructions to {output_file_path}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_assembly.py <input_json_file> <output_asm_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = extract_assembly_from_json(input_file, output_file)
    sys.exit(0 if success else 1) 