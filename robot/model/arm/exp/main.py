from robot.model.arm.exp.evaluate_model import evaluate_model

#evaluate_model('acrobat2', 'mlp', path='tmp', num_epoch=10, lr=0.001)
#evaluate_model('acrobat2', 'phys_gt', path='tmp_phys', num_epoch=10, weight_dq=0.0001, lr=0.001)

#evaluate_model('acrobat2', 'phys', path='tmp_phys', num_epoch=10, weight_dq=1., lr=0.001)

evaluate_model('arm', 'mlp', path='mlp_arm', num_epoch=100, weight_dq=1., lr=0.001)
