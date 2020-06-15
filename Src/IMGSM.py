import numpy as np
import torch
from misc_functions import preprocess_image, recreate_image, pil_loader
from PIL import Image
import torch.nn.functional as F
from torch import nn

class IterativeMaskedGradientSign():
    """
        iterative masked gradient sign method
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha

    def generate(self, original_image, target_class, itr_max, prob_threshold, mask, grad_type, verbose):
        # Targeting the specific class
        im_label_as_var = torch.from_numpy(np.asarray([target_class])).cuda()
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss().cuda()
        # Process image
        processed_image = preprocess_image(original_image).cuda()
        
        # if one-step attack can get very high attack confidence, and then
        # attack_cand is original input image
        attack_cand = original_image  # the candidate attack sample (PIL image)
        attack_prob = 0

        # Start iteration
        for itr in range(itr_max):
            
            self.model.zero_grad()
            # Forward pass
            processed_image.requires_grad = True
            output = self.model(processed_image)

            if verbose:
                init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                out = F.softmax(output, dim=1)
                confirmation_confidence = out[:,target_class].cpu().item()
                print('Iteration:', str(itr))
                print('pred_class: {}, pred_prob(target_class): {}.'.format(init_pred.cpu().item(), confirmation_confidence))
            
            # Calculate CE loss
            pred_loss = ce_loss(output, im_label_as_var)

            # Do backward pass
            pred_loss.backward()
            
            with torch.no_grad():
                # Create Noise
                if mask > 0.01:
                    if grad_type == 'random':
                        if verbose:
                            print('random imgsm')
                        grad_mask = torch.FloatTensor(processed_image.grad.shape).uniform_() > mask
                        adv_noise = self.alpha * torch.sign(processed_image.grad) * grad_mask.type(torch.float).cuda()
                    else:
                        if verbose:
                            print('topk imgsm')
                        K = int(round(processed_image.grad.numel()*(1.0 - mask)))
                        value, index = torch.topk(torch.abs(processed_image.grad).view(-1), k=K)                        
                        nindex = np.unravel_index(index.cpu().numpy(), processed_image.grad.shape)
                        grad_mask = torch.zeros(processed_image.grad.shape)
                        grad_mask[nindex[0],nindex[1],nindex[2],nindex[3]] = 1.0
                        adv_noise = self.alpha * torch.sign(processed_image.grad) * grad_mask.type(torch.float).cuda()
                else:
                    if verbose:
                        print('fgsm')
                    adv_noise = self.alpha * torch.sign(processed_image.grad)

                # Add noise to processed image
                processed_image -= adv_noise
                # Manually zero the gradients
                processed_image.grad.zero_()
            
            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image).cuda()
            # Forward pass
            with torch.no_grad():
                confirmation_out = self.model(prep_confirmation_image)
            # Get Probability
            output = F.softmax(confirmation_out, dim=1)
            confirmation_confidence = output[:,target_class].cpu().item()
            if confirmation_confidence > prob_threshold:
                if verbose:
                    print('pred_class: ', output.max(1, keepdim=True)[1].cpu().item())
                    print('predicted with confidence of:', confirmation_confidence)
                break
            else:
                attack_cand = recreated_image
                attack_prob = confirmation_confidence
                if verbose:
                    print('predicted with confidence of:', confirmation_confidence)

        return attack_cand, itr+1, attack_prob

def IMGSM(model, original_image, target_class, alpha, itr_max, prob_threshold, mask, grad_type='random', verbose=False):
    prob_max = prob_threshold[0]
    prob_min = prob_threshold[1]
    IMGS = IterativeMaskedGradientSign(model, alpha)
    current_mask = mask
    attack_prob = 0
    while attack_prob < prob_min and current_mask <= 0.998:
        recreated_image, itr, attack_prob = IMGS.generate(original_image, target_class, itr_max, prob_max, current_mask, grad_type, verbose)
        if current_mask < 0.99:
            current_mask += 0.01
        else:
            current_mask += 0.002

    return recreated_image, itr