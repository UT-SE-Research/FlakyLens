import torch
from captum.attr import IntegratedGradients

class CustomIntegratedGradients(IntegratedGradients):

    #def compute_gradients(self, inputs, target_ind):
    #    print("Executing custom compute_gradients method")  # Debugging line
    #    self.model.zero_grad()
    #    output = self.model(*inputs)
    #    selected = output[:, target_ind].sum()
    #    grads = torch.autograd.grad(selected, inputs, allow_unused=True)  # Added allow_unused=True
    #    return grads
    def __init__(self, model):
        super().__init__(model)
        self.model = model 

    def compute_gradients(self, inputs, target_ind):
        #print("Executing custom compute_gradients method")  # Debugging line
        self.model.zero_grad()
        embeddings, attention_mask = inputs  # Unpack inputs
        #embeddings = embeddings.float().requires_grad_(True)  # Ensure they are floating point and require gradients
        #attention_mask = attention_mask.float()  # Ensure attention_mask is also float
        embeddings = embeddings.clone().detach().requires_grad_(True).float()  # Ensure embeddings are float and have gradients

        # Enable gradients calculation
        torch.set_grad_enabled(True)
        #print(embeddings) 
        # Forward pass with the unpacked inputs
        output = self.model(embeddings, attention_mask, is_embedding=True)
        if isinstance(output, tuple):
            #print('tuple***')
            output = output[0]

        #print(f"Model output: {output}")  # Debugging line
        selected = output[:, target_ind].sum()
        #print(f"Selected output: {selected}")  # Debugging line

        grads = torch.autograd.grad(selected, embeddings, allow_unused=True)
        #self.model.eval()
        #print("Gradients calculated")  # Debugging line
        return grads

    def _attribute(self, inputs, target, additional_forward_args=None, **kwargs):
        #print("Custom _attribute method called")  # Debugging line
        target_ind = target if isinstance(target, int) else target[0]
        gradients = self.compute_gradients(inputs, target_ind)
        return gradients
        '''print("Custom _attribute method called")
        target_ind = target if isinstance(target, int) else target[0]

        # Implementing approximation similar to IntegratedGradients
        baseline = torch.zeros_like(inputs[0])  # Baseline, could be all zeros
        steps = 50  # Number of steps for the integral approximation
        scaled_inputs = [(baseline + float(i) / steps * (inputs[0] - baseline)) for i in range(steps + 1)]

        total_gradients = torch.zeros_like(inputs[0])
        for scaled_input in scaled_inputs:
            gradients = self.compute_gradients((scaled_input, inputs[1]), target_ind)
            total_gradients += gradients / steps

        # Final attributions are the product of the total gradients and the input difference
        attributions = (inputs[0] - baseline) * total_gradients

        # Compute approximation error
        output_original = self.model(inputs[0], inputs[1], is_embedding=True)[0, target_ind]
        output_baseline = self.model(baseline, inputs[1], is_embedding=True)[0, target_ind]
        approximation_error = (attributions.sum() - (output_original - output_baseline)).abs()

        return attributions, approximation_error'''

    def attribute(self, inputs, target=0, **kwargs):
        #print("Custom attribute method called")  # Debugging line
        return self._attribute(inputs, target, **kwargs)

    #def _attribute(self, inputs, target, additional_forward_args=None, **kwargs):
    #    print("Custom _attribute method called")  # Debugging line
    #    target_ind = target if isinstance(target, int) else target[0]
    #    gradients = self.compute_gradients(inputs, target_ind)
