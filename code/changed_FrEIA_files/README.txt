FrEIA code was changed to allow the use of recurrent subnetworks with GLOW with the required dimensions
in Pytorch LSTM API. 

coupling_layers.py:
changed forward function of "GLOWCouplingBlock" class to permute and concatenate the data correctly
for use with LSTM from pytorch

fixed_transforms.py:
added "recurrent=False" in forward function input variables of "PermuteRandom" class
to allow to propagate the variable "recurrent" through the whole net when using "model(x, c=y, recurrent=True)"

framework.py:
changed forward function of "ReversibleGraphNet" class by adding "recurrent=False" to input variables and
on line 380 and 382 added "recurrent=recurrent" to propagate the variable "recurrent" through the whole net
to allow using "model(x, c=y, recurrent=True)"