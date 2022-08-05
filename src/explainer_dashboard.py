import os.path
import shap
import pickle
import config
import socket
import inspect
from loguru import logger
from importlib import reload
from config import loging_config
from config import explainer_dashboard_path as paths
from PredictorPipeline.predicting.predictor import Predictor
from PredictorPipeline.evaluating.evaluator import Evaluator
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub


# https://explainerdashboard.readthedocs.io/en/latest/
# https://explainerdashboard.readthedocs.io/_/downloads/en/latest/pdf/
def run():

    reload(config)

    logger.configure(**loging_config)
    logger.info(f"start initializing '{__name__}' on '{socket.gethostname().upper()}'!")

    try:
        predictor = Predictor.load(path=f"{paths['input_folder']}/predictor.pkl")
        with open(os.path.join(config.evaluate_ml_models_path['output_folder'], 'labels.pkl'), 'rb') as handle:
            labels = pickle.load(handle)
    except (Exception, FileNotFoundError):
        logger.error(f"could not load predictor in {paths['input_folder']}/predictor.pkl")
        return None

    X = predictor.X_train
    y = predictor.y_train.astype(int)

    evaluator = Evaluator(predictor=predictor)
    df = evaluator.get_models(type="df")

    if isinstance(config.calculate_model_ids, str):
        model_ids = [config.calculate_model_ids]
    elif isinstance(config.calculate_model_ids, list):
        model_ids = config.calculate_model_ids
    else:
        logger.error(f"model_ids has to ne an instance of str or list")
        return None

    if not os.path.isdir(paths['output_folder']):
        try:
            os.mkdir(paths['output_folder'])
        except Exception:
            logger.error(f"Could not build {paths['output_folder']}!")
            return None

    for model_id in model_ids:

        # store dashboard
        folder = os.path.join(paths['output_folder'], model_id)
        if not os.path.isdir(folder):
            try:
                os.mkdir(folder)
            except (FileExistsError):
                pass
        else:
            if not config.recalculate:  # TODO: check if dashboard.yaml and explainer.dill are in directory
                continue

        model = evaluator.get_models(mode=model_id)

        if not model:
            os.rmdir(folder)
            logger.error(f"model with id {model_id} has not been calculated ...")
            continue

        estimator = model.get(list(model.keys())[0]).steps.pop()[1]
        X_trans = model.get(list(model.keys())[0]).transform(X)
        estimator.fit(X_trans, y)

        kwargs = _get_kwargs(ClassifierExplainer)

        explainer = ClassifierExplainer(estimator, X_trans, y,
                                        X_background=shap.sample(X_trans, 25),
                                        labels=list(labels.keys()),
                                        **kwargs
                                        )

        # TODO: add generic score (config.py)
        score = df.loc[df['id'] == model_id, 'mean_validation_accuracy'].values[0]

        db = ExplainerDashboard(explainer=explainer, title=f"{model_id}", name=f"{model_id}",
                                description=f"{str(estimator)}, mean_validation_accuracy: {round(score, 3)}")

        db.to_yaml(f"{os.path.join(folder, 'dashboard.yaml')}",
                   explainerfile=f"explainer.dill", dump_explainer=True)

    # run dashboard or hub
    dashboards = list()
    if config.run_dashboard == "all":
        for root, dirs, files in os.walk(paths['output_folder'], topdown=False):
            for name in dirs:
                dashboards.append(name)
    elif isinstance(config.run_dashboard, str):
        dashboards.append(config.run_dashboard)
    elif not isinstance(config.run_dashboard, list):
        logger.warning(f"run_dashboard is not instance of str (id), or list[ids] or 'all' ({config.run_dashboard})."
                       f"No dashboard will be started!")
        return None

    if len(dashboards) == 1:
        try:
            path = os.path.join(paths['output_folder'], dashboards[0])
            db = ExplainerDashboard.from_config(f"{os.path.join(path, 'explainer.dill')}",
                                                f"{os.path.join(path, 'dashboard.yaml')}")
            logger.info(f"... run loaded ExplainerDashboard [{dashboards[0]}]!")
            kwargs = _get_kwargs(db.run)
            db.run(**kwargs)
        except FileNotFoundError:
            logger.warning(f"did not found explainer.dill and/or dashboard.yaml in {path}!")
    else:
        list_db = []
        for id in dashboards:
            try:
                path = os.path.join(paths['output_folder'], id)
                db = ExplainerDashboard.from_config(
                    f"{os.path.join(path, 'explainer.dill')}",
                    f"{os.path.join(path, 'dashboard.yaml')}",
                )
                list_db.append(db)
            except FileNotFoundError:
                logger.warning(f"did not found explainer.dill and/or dashboard.yaml in {path}!")

        logger.info(f"... run ExplainerHub with dashboards: {list_db}")

        hub = ExplainerHub(list_db)
        kwargs = _get_kwargs(hub.run)
        hub.run(**kwargs)


def _get_kwargs(method):
    """ gets arguments as dictionary for the according method from config.explainer_args """

    try:
        from config import explainer_args as kwargs
    except Exception:
        logger.warning(f"no explainer_args to import! Default values will be used!")
        return {}

    signature = inspect.signature(method)
    params = list(signature.parameters)

    kwargs = {k: v for k, v in kwargs.items() if k in params}

    return kwargs


if __name__ == '__main__':
    run()
