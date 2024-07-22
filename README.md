# Laplacian Pyramid

This is an implementation of the Laplacian Pyramid 
encoding algorithm for images, as described in the 
1983 paper _The Laplacian Pyramid as a Compact Image
Code_ by Peter J. Burt & Edward H. Adelson. It is
also utilized to implement image blending.

## Background

The Laplacian Pyramid is a decomposition of the 
information captured within an image, across
different resolutions. Hence - different layers
represent different levels of _visual detail_. Once
extracted, the layers of information in different 
resolutions can be used to reconstruct the original 
image.

This encoding scheme provides us with access to the 
different levels of visual detail within an image,
which opens the door to some interesting applications,
some of which are:

* Compression: since most of visual information is 
usually low-resolution, the high-resolution layers
of information are rather spurious - and can be
effectively compressed.
* Blending: we may selectively combine the visual
information at different levels of resolutions, to
blend images together.