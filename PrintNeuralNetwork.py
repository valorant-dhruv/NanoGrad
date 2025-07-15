import matplotlib.pyplot as plt
import numpy as np

def visualize_neural_network(neural_network, inputs=None, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get layer sizes
    layer_sizes = neural_network.layer_sizes
    num_layers = len(layer_sizes)
    
    # Calculate positions for each neuron
    max_neurons = max(layer_sizes)
    layer_positions = {}
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        x = layer_idx * 3  # Horizontal spacing between layers
        
        # Center neurons vertically
        if layer_size == 1:
            y_positions = [0]
        else:
            y_positions = np.linspace(-(layer_size-1)/2, (layer_size-1)/2, layer_size)
        
        layer_positions[layer_idx] = [(x, y) for y in y_positions]
    
    # Draw connections and weights
    for layer_idx in range(1, num_layers):
        current_layer = neural_network.layers[layer_idx]
        prev_layer_positions = layer_positions[layer_idx - 1]
        curr_layer_positions = layer_positions[layer_idx]
        
        for neuron_idx, neuron in enumerate(current_layer.neurons):
            curr_x, curr_y = curr_layer_positions[neuron_idx]
            
            # Draw connections from previous layer
            for prev_neuron_idx, (prev_x, prev_y) in enumerate(prev_layer_positions):
                # Get weight value
                weight = neuron.weights[prev_neuron_idx].value
                
                # Draw connection line
                ax.plot([prev_x, curr_x], [prev_y, curr_y], 'k-', alpha=0.6, linewidth=1)
                
                # Calculate offset position for weight label to avoid overlap
                mid_x = (prev_x + curr_x) / 2
                mid_y = (prev_y + curr_y) / 2
                
                # Add small offset based on connection direction to reduce overlap
                offset_x = 0.1 * (prev_neuron_idx - len(prev_layer_positions)/2) * 0.2
                offset_y = 0.1 * (neuron_idx - len(current_layer.neurons)/2) * 0.2
                
                label_x = mid_x + offset_x
                label_y = mid_y + offset_y
                
                # Create weight label with smaller font and tighter spacing
                ax.text(label_x, label_y, f'{weight:.2f}', 
                       fontsize=7, ha='center', va='center', color='black',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                               edgecolor='gray', alpha=0.9, linewidth=0.5))
    
    # Draw neurons and biases
    for layer_idx, positions in layer_positions.items():
        layer = neural_network.layers[layer_idx]
        
        for neuron_idx, (x, y) in enumerate(positions):
            # Draw neuron circle
            circle = plt.Circle((x, y), 0.3, color='white', ec='black', linewidth=2)
            ax.add_patch(circle)
            
            # Add neuron value inside the circle
            if layer_idx == 0:
                # For input layer, show the actual input values if provided, otherwise show neuron values
                if inputs is not None:
                    neuron_value = inputs[neuron_idx]
                else:
                    neuron = layer.neurons[neuron_idx]
                    neuron_value = neuron.neuron_value.value
            else:
                # For other layers, show the computed neuron values
                neuron = layer.neurons[neuron_idx]
                neuron_value = neuron.neuron_value.value
            
            ax.text(x, y, f'{neuron_value:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='black')
            
            # Add bias above the neuron (skip input layer as it typically has no bias)
            if layer_idx > 0:
                neuron = layer.neurons[neuron_idx]
                bias_value = neuron.bias.value
                ax.text(x, y + 0.6, f'b={bias_value:.2f}', 
                       ha='center', va='center', fontsize=8, color='black',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='lightblue', 
                               edgecolor='black', alpha=0.8, linewidth=0.5))
    
    # Add layer labels
    for layer_idx in range(num_layers):
        x = layer_idx * 3
        y = max(layer_sizes) / 2 + 1.2
        
        if layer_idx == 0:
            layer_name = "Input Layer"
        elif layer_idx == num_layers - 1:
            layer_name = "Output Layer"
        else:
            layer_name = f"Hidden Layer {layer_idx}"
            
        ax.text(x, y, layer_name, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                       edgecolor='black', alpha=0.8))
    
    # Set axis properties with more space
    ax.set_xlim(-0.7, (num_layers - 1) * 3 + 0.7)
    ax.set_ylim(-max(layer_sizes)/2 - 1.5, max(layer_sizes)/2 + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('Neural Network Visualization\n', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


# Alternative version with even better spacing for complex networks
def visualize_neural_network_clean(neural_network, inputs=None, figsize=(14, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get layer sizes
    layer_sizes = neural_network.layer_sizes
    num_layers = len(layer_sizes)
    
    # Calculate positions for each neuron with more spacing
    layer_positions = {}
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        x = layer_idx * 4  # Increased horizontal spacing
        
        # Center neurons vertically with more spacing
        if layer_size == 1:
            y_positions = [0]
        else:
            y_positions = np.linspace(-(layer_size-1)*0.8, (layer_size-1)*0.8, layer_size)
        
        layer_positions[layer_idx] = [(x, y) for y in y_positions]
    
    # Draw connections first (behind neurons)
    for layer_idx in range(1, num_layers):
        current_layer = neural_network.layers[layer_idx]
        prev_layer_positions = layer_positions[layer_idx - 1]
        curr_layer_positions = layer_positions[layer_idx]
        
        for neuron_idx, neuron in enumerate(current_layer.neurons):
            curr_x, curr_y = curr_layer_positions[neuron_idx]
            
            # Draw connections from previous layer
            for prev_neuron_idx, (prev_x, prev_y) in enumerate(prev_layer_positions):
                # Get weight value
                weight = neuron.weights[prev_neuron_idx].value
                
                # Draw connection line with thickness based on weight magnitude
                line_width = min(max(abs(weight) * 2, 0.5), 3)
                color = 'red' if weight < 0 else 'blue'
                
                ax.plot([prev_x, curr_x], [prev_y, curr_y], color=color, 
                       alpha=0.6, linewidth=line_width)
    
    # Draw weight labels with better positioning
    for layer_idx in range(1, num_layers):
        current_layer = neural_network.layers[layer_idx]
        prev_layer_positions = layer_positions[layer_idx - 1]
        curr_layer_positions = layer_positions[layer_idx]
        
        # Calculate a grid for weight labels to avoid overlap
        prev_layer_size = len(prev_layer_positions)
        curr_layer_size = len(curr_layer_positions)
        
        for neuron_idx, neuron in enumerate(current_layer.neurons):
            curr_x, curr_y = curr_layer_positions[neuron_idx]
            
            for prev_neuron_idx, (prev_x, prev_y) in enumerate(prev_layer_positions):
                weight = neuron.weights[prev_neuron_idx].value
                
                # Position weight label closer to source neuron to reduce overlap
                label_x = prev_x + 0.7
                label_y = prev_y + (neuron_idx - curr_layer_size/2) * 0.3
                
                ax.text(label_x, label_y, f'{weight:.2f}', 
                       fontsize=7, ha='center', va='center', color='black',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                               edgecolor='gray', alpha=0.9, linewidth=0.5))
    
    # Draw neurons and biases
    for layer_idx, positions in layer_positions.items():
        layer = neural_network.layers[layer_idx]
        
        for neuron_idx, (x, y) in enumerate(positions):
            # Draw neuron circle
            circle = plt.Circle((x, y), 0.35, color='white', ec='black', linewidth=2)
            ax.add_patch(circle)
            
            # Add neuron value inside the circle
            if layer_idx == 0:
                if inputs is not None:
                    neuron_value = inputs[neuron_idx]
                else:
                    neuron = layer.neurons[neuron_idx]
                    neuron_value = neuron.neuron_value.value
            else:
                neuron = layer.neurons[neuron_idx]
                neuron_value = neuron.neuron_value.value
            
            ax.text(x, y, f'{neuron_value:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='black')
            
            # Add bias above the neuron
            if layer_idx > 0:
                neuron = layer.neurons[neuron_idx]
                bias_value = neuron.bias.value
                ax.text(x, y + 0.7, f'b={bias_value:.2f}', 
                       ha='center', va='center', fontsize=8, color='black',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='lightblue', 
                               edgecolor='black', alpha=0.8))
    
    # Add layer labels
    for layer_idx in range(num_layers):
        x = layer_idx * 4
        y = max([max(layer_sizes) * 0.8, 3]) + 1.5
        
        if layer_idx == 0:
            layer_name = "Input Layer"
        elif layer_idx == num_layers - 1:
            layer_name = "Output Layer"
        else:
            layer_name = f"Hidden Layer {layer_idx}"
            
        ax.text(x, y, layer_name, ha='center', va='center', 
               fontsize=12, fontweight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                       edgecolor='black', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(-1, (num_layers - 1) * 4 + 1)
    ax.set_ylim(-max(layer_sizes) - 1, max(layer_sizes) + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('Neural Network Visualization\n(Blue=Positive weights, Red=Negative weights)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
