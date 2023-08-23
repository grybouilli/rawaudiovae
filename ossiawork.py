from ossiapreload import *

def raw_to_z_dist(test_dataloader, raw_model, device):
    init_test = True
    for iterno, test_sample in enumerate(test_dataloader):
        with torch.no_grad():
            test_sample = test_sample.to(device)
            test_mu, test_logvar = raw_model.encode(test_sample)

        if init_test:
            test_z_mu = test_mu 
            test_z_logvar = test_logvar
            init_test = False

        else:
            test_z_mu = torch.cat((test_z_mu, test_mu ),0)
            test_z_logvar = torch.cat((test_z_logvar, test_logvar ),0)
    return test_z_mu, test_z_logvar

test1_z_mu, test1_z_logvar = raw_to_z_dist(test_dataloader1, model, device)
test2_z_mu, test2_z_logvar = raw_to_z_dist(test_dataloader2, model, device)

def raw_interpolate_stepwise_z_dist(test1_z_mu, test1_z_logvar, test2_z_mu, test2_z_logvar, interpolation_range, raw_model):

    init_test = True
    for interpolation in interpolation_range:

        inter_z_mu = torch.add( torch.mul(test1_z_mu, (1-interpolation)), torch.mul(test2_z_mu, interpolation) )
        inter_z_logvar = torch.add( torch.mul(test1_z_logvar, (1-interpolation)), torch.mul(test2_z_logvar, interpolation) )
        
        with torch.no_grad():
            test_pred_z = raw_model.reparameterize(inter_z_mu, inter_z_logvar)
            test_pred = raw_model.decode(test_pred_z)

        if init_test:
            test_predictions = test_pred
            init_test = False

        else:
            test_predictions = torch.cat((test_predictions, test_pred ),0)
        
    return test_predictions

interpolation_range = np.arange(0, 1.1, 0.2)

inter_raw_all = raw_interpolate_stepwise_z_dist(test1_z_mu, test1_z_logvar, test2_z_mu, test2_z_logvar, interpolation_range, model)

output = inter_raw_all.view(-1).cpu().numpy()

