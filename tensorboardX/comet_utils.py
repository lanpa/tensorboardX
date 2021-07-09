import logging
import json
import functools
from io import BytesIO
import numpy as np
from .summary import _clean_tag
try:
    import comet_ml
    comet_installed = True
    from PIL import Image
except ImportError:
    comet_installed = False


class CometLogger:
    def __init__(self, comet_config={"disabled": True}):
        global comet_installed
        self._logging = None
        self._comet_config = comet_config
        if comet_config["disabled"] is True:
            self._logging = False
        elif comet_config["disabled"] is False and comet_installed is False:
            raise Exception("Comet not installed. Run 'pip install comet-ml'")

    def _requiresComet(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self = args[0]
            global comet_installed
            if self._logging is None and comet_installed:
                self._logging = False
                try:
                    if 'api_key' not in self._comet_config.keys():
                        comet_ml.init()
                    if comet_ml.get_global_experiment() is not None:
                        logging.warning("You have already created a comet \
                                        experiment manually, which might \
                                        cause clashes")
                    self._experiment = comet_ml.Experiment(**self._comet_config)
                    self._logging = True
                    self._experiment.log_other("Created from", "tensorboardX")
                except Exception as e:
                    logging.warning(e)

            if self._logging is True:
                return method(*args, **kwargs)
        return wrapper

    @_requiresComet
    def end(self):
        """Ends an experiment."""
        self._experiment.end()
        comet_ml.config.experiment = None

    @_requiresComet
    def log_metric(self, tag, display_name, value, step=None, epoch=None,
                   include_context=True):
        """Logs a general metric (i.e accuracy, f1)..

        Args:
            tag: String - Data identifier
            display_name: The title of the plot. If empty string is passed,
              `tag` will be used.
            value: Float/Integer/Boolean/String
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
            include_context: Optional. If set to True (the default),
                the current context will be logged along the metric.
        """
        name = _clean_tag(tag) if display_name == "" else display_name
        self._experiment.log_metric(name, value, step, epoch,
                                    include_context)

    @_requiresComet
    def log_metrics(self, dic, prefix=None, step=None, epoch=None):
        """Logs a key,value dictionary of metrics.

        Args:
            dic: key,value dictionary of metrics
            prefix: prefix added to metric name
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
        """
        self._experiment.log_metrics(dic, prefix, step, epoch)

    @_requiresComet
    def log_parameters(self, parameters, prefix=None, step=None):
        """Logs a dictionary (or dictionary-like object) of multiple parameters.

        Args:
            parameters: key,value dictionary of parameters
            prefix: prefix added to metric name
            step: Optional. Used as the X axis when plotting on comet.ml
        """
        self._experiment.log_parameters(parameters, prefix, step)

    @_requiresComet
    def log_audio(self, audio_data, sample_rate=None, file_name=None,
                  metadata=None, overwrite=False, copy_to_tmp=True,
                  step=None):
        """Logs the audio Asset determined by audio data.

        Args:
        audio_data: String or a numpy array - either the file path
            of the file you want to log, or a numpy array given to
            scipy.io.wavfile.write for wav conversion.
        sample_rate: Integer - Optional. The sampling rate given to
            scipy.io.wavfile.write for creating the wav file.
        file_name: String - Optional. A custom file name to be displayed.
            If not provided, the filename from the audio_data argument
            will be used.
        metadata: Some additional data to attach to the the audio asset.
            Must be a JSON-encodable dict.
        overwrite: if True will overwrite all existing assets with the same name.
        copy_to_tmp: If audio_data is a numpy array, then this flag
            determines if the WAV file is first copied to a temporary
            file before upload. If copy_to_tmp is False, then it is sent
            directly to the cloud.
        step: Optional. Used to associate the audio asset to a specific step.
        """
        self._experiment.log_audio(audio_data, sample_rate, file_name,
                                   metadata, overwrite, copy_to_tmp,
                                   step)

    @_requiresComet
    def log_text(self, text, step=None, metadata=None):
        """Logs the text. These strings appear on the Text Tab in the Comet UI.

        Args:
        text: string to be stored
        step: Optional. Used to associate the asset to a specific step.
        metadata: Some additional data to attach to the the text. Must
            be a JSON-encodable dict.
        """
        self._experiment.log_text(text, step, metadata)

    @_requiresComet
    def log_histogram(self, values, name=None, step=None, epoch=None,
                      metadata=None, **kwargs):
        """Logs a histogram of values for a 3D chart as an asset for
           this experiment. Calling this method multiple times with the
           same name and incremented steps will add additional histograms
           to the 3D chart on Comet.ml.

        Args:
        values: a list, tuple, array (any shape) to summarize, or a
            Histogram object
        name: str (optional), name of summary
        step: Optional. Used as the Z axis when plotting on Comet.ml.
        epoch: Optional. Used as the Z axis when plotting on Comet.ml.
        metadata: Optional: Used for items like prefix for histogram name.
        kwargs: Optional. Additional keyword arguments for histogram.
        """
        self._experiment.log_histogram_3d(values, name, step,
                                          epoch, metadata,
                                          **kwargs)

    @_requiresComet
    def log_curve(self, name, x, y, overwrite=False, step=None):
        """Log timeseries data.

        Args:
        name: (str) name of data
        x: array of x-axis values
        y: array of y-axis values
        overwrite: (optional, bool) if True, overwrite previous log
        step: (optional, int) the step value
        """
        self._experiment.log_curve(name, x.tolist(), y.tolist(), overwrite, step)

    @_requiresComet
    def log_image_encoded(self, encoded_image_string, tag, step=None):
        """Logs the image. Images are displayed on the Graphics tab on Comet.ml.

        Args:
        encoded_image_string: Required. An encoded image string
        tag: String - Data identifier
        step: Optional. Used to associate the image asset to a specific step.
        """
        buff = BytesIO(encoded_image_string)
        image_pil = Image.open(buff)
        name = _clean_tag(tag)
        self._experiment.log_image(image_pil, name, step=step)

    @_requiresComet
    def log_asset(self, file_data, file_name=None, overwrite=False,
                  copy_to_tmp=True, step=None, metadata=None):
        """Logs the Asset determined by file_data.

        Args:
        file_data: String or File-like - either the file path of the
            file you want to log, or a file-like asset.
        file_name: String - Optional. A custom file name to be displayed.
            If not provided the filename from the file_data argument will be used.
        overwrite: if True will overwrite all existing assets with
            the same name.
        copy_to_tmp: If file_data is a file-like object, then this flag
            determines if the file is first copied to a temporary file
            before upload. If copy_to_tmp is False, then it is sent
            directly to the cloud.
        step: Optional. Used to associate the asset to a specific step.
        """
        self._experiment.log_asset(file_data, file_name, overwrite,
                                   copy_to_tmp, step, metadata)

    @_requiresComet
    def log_asset_data(self, data, name=None, overwrite=False, step=None,
                       metadata=None, epoch=None):
        """Logs the data given (str, binary, or JSON).

        Args:
        data: data to be saved as asset
        name: String, optional. A custom file name to be displayed If
            not provided the filename from the temporary saved file
            will be used.
        overwrite: Boolean, optional. Default False. If True will
            overwrite all existing assets with the same name.
        step: Optional. Used to associate the asset to a specific step.
        epoch: Optional. Used to associate the asset to a specific epoch.
        metadata: Optional. Some additional data to attach to the
            asset data. Must be a JSON-encodable dict.
        """
        self._experiment.log_asset_data(data, name, overwrite, step,
                                        metadata, epoch)

    @_requiresComet
    def log_embedding(self, vectors, labels, image_data=None,
                      image_preprocess_function=None, image_transparent_color=None,
                      image_background_color_function=None, title="Comet Embedding",
                      template_filename=None,
                      group=None):
        """Log a multi-dimensional dataset and metadata for viewing
           with Comet's Embedding Projector (experimental).

        Args:
        vectors: the tensors to visualize in 3D
        labels: labels for each tensor
        image_data: (optional) list of arrays or Images
        image_preprocess_function: (optional) if image_data is an array,
            apply this function to each element first
        image_transparent_color: a (red, green, blue) tuple
        image_background_color_function: a function that takes an
            index, and returns a (red, green, blue) color tuple
        title: (optional) name of tensor
        template_filename: (optional) name of template JSON file
        """
        image_size = None
        if labels is None:
            return
        if image_data is not None:
            image_data = image_data.cpu().detach().numpy()
            image_size = image_data.shape[1:]
            if image_size[0] == 1:
                image_size = image_size[1:]
        if type(labels) == list:
            labels = np.array(labels)
        else:
            labels = labels.cpu().detach().numpy()
        self._experiment.log_embedding(vectors, labels, image_data,
                                       image_size, image_preprocess_function,
                                       image_transparent_color,
                                       image_background_color_function,
                                       title, template_filename,
                                       group)

    @_requiresComet
    def log_mesh(self, tag, vertices, colors, faces, config_dict, step, walltime):
        """Logs a mesh as an asset

        Args:
        tag: Data identifier
        vertices: List of the 3D coordinates of vertices.
        colors: Colors for each vertex
        faces: Indices of vertices within each triangle.
        config_dict: Dictionary with ThreeJS classes names and configuration.
        step: step value to record
        walltime: Optional override default walltime (time.time())
            seconds after epoch of event
        """
        mesh_json = {}
        mesh_json['tag'] = tag
        mesh_json['vertices'] = vertices.tolist()
        mesh_json['colors'] = colors.tolist()
        mesh_json['faces'] = faces.tolist()
        mesh_json['config_dict'] = config_dict
        mesh_json['walltime'] = walltime
        mesh_json['asset_type'] = 'mesh'
        mesh_json = json.dumps(mesh_json)
        self.log_asset_data(mesh_json, tag, step=step)

    @_requiresComet
    def log_raw_figure(self, tag, asset_type, step=None, **kwargs):
        """Logs a histogram as an asset.

        Args:
        tag: Data identifier
        asset_type: List of the 3D coordinates of vertices.
        step: step value to record
        """
        file_json = kwargs
        file_json['asset_type'] = asset_type
        self.log_asset_data(file_json, tag, step=step)
