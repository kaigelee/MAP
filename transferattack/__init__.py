import importlib

attack_zoo = {
    # gradient
    'fgsm': ('.gradient.fgsm', 'FGSM'),
    'ifgsm': ('.gradient.ifgsm', 'IFGSM'),
    'mifgsm': ('.gradient.mifgsm', 'MIFGSM'),
    # input transformation
    ## Untargeted
    'dim': ('.input_transformation.dim', 'DIM'),    
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise Exception('Unspported attack algorithm {}'.format(attack_name))
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)
    return attack_class


__version__ = '1.0.0'
