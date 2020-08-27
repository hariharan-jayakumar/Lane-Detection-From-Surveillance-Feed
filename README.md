## Background
During the period of November 2018-January 2019, I interned in the Hardware and Embedded Systems Lab at NTU, Singapore. Under the esteemed guidance of Dr Nirmala Radhkrishnan and Professor Thambipillai Srikanthan, I worked on building a tool to automatically detect and mark lanes in roads based on feed from surveillance cameras. Although the dataset I worked with is proprietory, the script I wrote is available here.

## Nature of Data
The data-set consisted of videos of elevated roads taken from an elevated angle. All videos were taken in day-light in varying contrast and brightness. Some videos were taken at signals with higher traffic, whereas some videos were taken in free-ways. A notable characteristic in some videos were the bus lanes, with the buses stopping for quite a long period of time.  

## Methodology
Due to the scarcity of data available and because we wanted the inferencing to be done on the edge, we decided to leverage traditional computer vision concepts to mark lanes.
I employed Computer Vision concepts to use the shape of the lanes, the colour of the lanes, mapping the direction of traffic with the direction of the lanes, vanishing point gradient and foreground subtraction.

## Results
For a majority of video samples given, we managed to identify the lanes accurately - 112 out of 156 video angles. The results, however, can not be shared since the data-set is propreitory.

Some areas we couldn't manage to achieve satisfactory results were in the case of intersections since the nature of traffic and the lanes were not in a straight line.
