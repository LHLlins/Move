  *	???S?ye@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenatea?$?Ӳ?!?L?afE@)=a??M??1??ؗ??C@:Preprocessing2U
Iterator::Model::ParallelMapV2	?<??t??!qQ?k?4@)	?<??t??1qQ?k?4@:Preprocessing2F
Iterator::Modelu??.???!k?C??5B@)<?)??1?RU1K?.@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?K??!Wd??Ş+@)?5?;Nё?1Il  KA$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??z????!??}`?O@):y?	?5??1h?$??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!O!W?y?!;?GR?u@)!O!W?y?1;?GR?u@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[1]::FromTensor4?ތ??r?!?ib >@)4?ތ??r?1?ib >@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??@?M???!???7LF@)?vöEi?1U?0?ź??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[0]::TensorSlice?e3??V?!2?t?~???)?e3??V?12?t?~???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.