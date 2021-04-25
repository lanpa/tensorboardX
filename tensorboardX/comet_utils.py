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
                    logging.warning("You have already created a comet \
                                    experiment manually, which might \
                                    cause clashes")
                self._experiment = comet_ml.Experiment()
                self._logging = True
            except Exception as e:
                logging.warning(e)


    def set_model_graph(self, graph):
        """Sets the current experiment computation graph.

        Args:
            graph: String or Google Tensorflow Graph Format
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.set_model_graph(graph)

    def end(self):
        """Ends an experiment."""
        if self._logging:
            self._experiment.end()
            comet_ml.config.experiment = None

    def log_metric(self, name, value, step=None, epoch=None,
                   include_context=True):
        """Logs a general metric (i.e accuracy, f1)..

        Args:
            name: String - name of your metric
            value: Float/Integer/Boolean/String
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
            include_context: Optional. If set to True (the default),
                the current context will be logged along the metric.
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_metric(name, value, step, epoch,
                                        include_context)

    def log_parameter(self, name, value, step=None):
        """Logs a single hyperparameter.

        Args:
            name: String - name of your parameter
            value: Float/Integer/Boolean/String
            step: Optional. Used as the X axis when plotting on comet.ml
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_parameter(name, value, step)

    def log_metrics(self, dic, prefix=None, step=None, epoch=None):
        """Logs a key,value dictionary of metrics.

        Args:
            dic: key,value dictionary of metrics
            prefix: predfix added to metric name
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_metrics(dic, prefix, step, epoch)

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
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_audio(audio_data, sample_rate, file_name,
                                       metadata, overwrite, copy_to_tmp,
                                       step)
        
    def log_text(self, text, step=None, metadata=None):
        """Logs the text. These strings appear on the Text Tab in the Comet UI.

        Args:  
        text: string to be stored
        step: Optional. Used to associate the asset to a specific step.
        metadata: Some additional data to attach to the the text. Must
            be a JSON-encodable dict.
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_text(text, step, metadata)

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
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_histogram_3d(values, name, step,
                                              epoch, metadata,
                                              **kwargs)

    def log_curve(self,  name, x, y, overwrite=False, step=None):
        """Log timeseries data.

        Args:  
        name: (str) name of data
        x: list of x-axis values
        y: list of y-axis values
        overwrite: (optional, bool) if True, overwrite previous log
        step: (optional, int) the step value
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_curve(name, x, y, overwrite, step)

    def log_model(self, name, file_or_folder, file_name=None,
                  overwrite=False, metadata=None, copy_to_tmp=True,
                  prepend_folder_name=True):
        """Logs the model data under the name. Data can be a file path,
           a folder path or a file-like object.

        Args:  
        name: string (required), the name of the model
        file_or_folder: the model data (required); can be a file path,
            a folder path or a file-like object.
        file_name: (optional) the name of the model data. Used with
            file-like objects or files only.
        overwrite: boolean, if True, then overwrite previous versions
            Does not apply to folders.
        metadata: Some additional data to attach to the the data. Must
            be a JSON-encodable dict.
        copy_to_tmp: for file name or file-like; if True copy to temporary
            location before uploading; if False, then upload from current location
        prepend_folder_name: boolean, default True. If True and logging
            a folder, prepend file path by the folder name.
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_model(name, file_or_folder, file_name,
                                       overwrite, metadata, copy_to_tmp,
                                       prepend_folder_name)

    def log_image(self, image_data, name=None, overwrite=False,
                  image_format="png", image_scale=1.0, image_shape=None,
                  image_colormap=None, image_minmax=None, image_channels="last",
                  copy_to_tmp=True, step=None):
        """Logs the image. Images are displayed on the Graphics tab on Comet.ml.

        Args:  
        image_data: Required. image_data is one of the following:
            a path (string) to an image
            a file-like object containing an image
            a numpy matrix
            a TensorFlow tensor
            a PyTorch tensor
            a list or tuple of values
            a PIL Image
        name: String - Optional. A custom name to be displayed on the
            dashboard. If not provided the filename from the image_data
            argument will be used if it is a path.
        overwrite: Optional. Boolean - If another image with the same
            name exists, it will be overwritten if overwrite is set to True.
        image_format: Optional. String. Default: 'png'. If the image_data
            is actually something that can be turned into an image, this
            is the format used. Typical values include 'png' and 'jpg'.
        image_scale: Optional. Float. Default: 1.0. If the image_data
            is actually something that can be turned into an image,
            this will be the new scale of the image.
        image_shape: Optional. Tuple. Default: None. If the image_data
            is actually something that can be turned into an image,
            this is the new shape of the array. Dimensions are (width, height)
            or (width, height, colors) where colors is 3 (RGB) or 1 (grayscale).
        image_colormap: Optional. String. If the image_data is actually
            something that can be turned into an image, this is the
            colormap used to colorize the matrix.
        image_minmax: Optional. (Number, Number). If the image_data 
            is actually something that can be turned into an image,
            this is the (min, max) used to scale the values. Otherwise,
            the image is autoscaled between (array.min, array.max).
        image_channels: Optional. Default 'last'. If the image_data is
            actually something that can be turned into an image, this is
            the setting that indicates where the color information is in
            the format of the 2D data. 'last' indicates that the data is in
            (rows, columns, channels) where 'first' indicates (channels, rows, columns).
        copy_to_tmp: If image_data is not a file path, then this flag
            determines if the image is first copied to a temporary
            file before upload. If copy_to_tmp is False, then it is sent
            directly to the cloud.
        step: Optional. Used to associate the image asset to a specific step.
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_image(image_data, name, overwrite,
                                       image_format, image_scale,
                                       image_shape, image_colormap,
                                       image_minmax, image_channels,
                                       copy_to_tmp, step)

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
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_asset(file_data, file_name, overwrite,
                                       copy_to_tmp, step, metadata)

    def log_asset_data(self, data, name=None, overwrite=False, step=None,
                       metadata=None, file_name=None, epoch=None):
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
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_asset_data(data, name, overwrite, step,
                                            metadata, file_name, epoch)

    def log_embedding(self, vectors, labels, image_data=None, image_size=None,
                      image_preprocess_function=None, image_transparent_color=None,
                      image_background_color_function=None, title="Comet Embedding",
                      template_filename="template_projector_config.json",
                      group=None):
        """Log a multi-dimensional dataset and metadata for viewing
           with Comet's Embedding Projector (experimental).

        Args:  
        vectors: the tensors to visualize in 3D
        labels: labels for each tensor
        image_data: (optional) list of arrays or Images
        image_size: (optional, required if image_data is given) the
            size of each image
        image_preprocess_function: (optional) if image_data is an array,
            apply this function to each element first
        image_transparent_color: a (red, green, blue) tuple
        image_background_color_function: a function that takes an
            index, and returns a (red, green, blue) color tuple
        title: (optional) name of tensor
        template_filename: (optional) name of template JSON file
        """
        self.__start_experiment__()
        if self._logging:
            self._experiment.log_embedding(vectors, labels, image_data,
                                           image_size, image_preprocess_function,
                                           image_transparent_color,
                                           image_background_color_function,
                                           title, template_filename,
                                           group)
       