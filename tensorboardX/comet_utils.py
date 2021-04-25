import logging
try:
    import comet_ml
    comet_installed = True
except:
    comet_installed = False


class CometLogger():
    def __init__(self):
        self._logging = None
    
    def __start_experiment__(self):
        global comet_installed
        if self._logging is not None:
            return
        elif comet_installed:
            self._logging = False
            try:
                if comet_ml.get_global_experiment() is not None:
                    logging.warning("DOUBLE LOGGING Warning")
                self._experiment = comet_ml.Experiment()
                self._logging = True
            except:
                logging.warning("COMET_API_KEY Warning")

    def set_model_graph(self, graph):
        self.__start_experiment__()
        if self._logging:
            self._experiment.set_model_graph(graph)

    def end(self):
        self._experiment.end()

    def log_metric(self, name, value, step, epoch=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_metric(name, value, step, epoch)

    def log_parameter(self, name, value, step=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_parameter(name, value, step)

    def log_metrics(self, dict_, step, epoch=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_metrics(dict_, step=step, epoch=epoch)

    def log_audio(self, audio_data, sample_rate=None, file_name=None,
                  step=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_audio(audio_data, sample_rate, file_name,
                                       step=step)
        
    def log_text(self, text, step=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_text(text, step)

    def log_histogram(self, values, name=None, step=None, epoch=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_histogram_3d(values, name, step, epoch)

    def log_curve(self, name, x, y, step=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_curve(name, x, y, step=step)

    def log_model(self, path):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_model("Model", path)

    def log_image(self, image_data, name=None, step=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_image(image_data, name, step=step)

    def log_asset(self, file_data, file_name=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_asset(file_data, file_name)

    def log_asset_data(self, data, name=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_asset(data, name)

    def log_embedding(self, vectors, labels, image_data=None):
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_embedding(vectors, labels, image_data)
       