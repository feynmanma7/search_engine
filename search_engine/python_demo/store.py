import pickle
import os


def save_model(model_dir_path=None, model=None):
    for model_name, model_value in model.items():
        model_path = os.path.join(model_dir_path, model_name + '.pkl')
        with open(model_path, 'wb') as fw:
            pickle.dump(model_value, fw)


def load_model(model_dir_path=None):
    model = {}
    for model_path in os.listdir(model_dir_path):
        model_path = os.path.join(model_dir_path, model_path)
        if not os.path.exists(model_path):
            continue

        with open(model_path, 'rb') as fr:
            model_value = pickle.load(fr)
            model_name = model_path.split('/')[-1].split('.')[0]
            model[model_name] = model_value

    return model