# Global variables to store normalization parameters
output_min = None
output_max = None

def normalise_input(inputs):
    normalized = []
    for input_row in inputs:
        # Normalize parameters (0-10 range) and rows (0-500000 range)
        norm_params = input_row[0] / 10.0
        norm_rows = input_row[1] / 500000.0
        normalized.append([norm_params, norm_rows])
    return normalized

def normalise_output(actual_output):
    global output_min, output_max
    
    # Store min/max values for denormalization
    output_min = min(actual_output)
    output_max = max(actual_output)
    
    # Create normalized copy (don't modify original)
    normalized = []
    for value in actual_output:
        normalized_value = (value - output_min) / (output_max - output_min)
        normalized.append(normalized_value)
    
    return normalized

def denormalise_output(normalized_value):
    global output_min, output_max
    
    if output_min is None or output_max is None:
        raise ValueError("Must call normalise_output first to set min/max values")
    
    # Convert back to original scale
    if hasattr(normalized_value, 'value'):
        # Handle Parameter objects
        original_value = normalized_value.value * (output_max - output_min) + output_min
    else:
        # Handle regular numbers
        original_value = normalized_value * (output_max - output_min) + output_min
    
    return original_value


#We use mean squared error as a loss function
def mean_squared_error(actual_output, predicted_output):
  final_loss = 0
  for actual, predicted in zip(actual_output, predicted_output):
    loss = (actual - predicted)**2 / len(actual_output)
    final_loss += loss
  return final_loss
