#-------------------------------------------------
#
# Project created by QtCreator 2017-07-13T04:02:37
#
#-------------------------------------------------

QT       -= core

QT       -= gui

TARGET = FaceRecogtion
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    Utils.cpp \
    detectFace.cpp \
    preprocessface.cpp \
    Recongnition.cpp

INCLUDEPATH += /usr/local/include  \
                /usr/local/include/opencv  \
                /usr/local/include/opencv2 \


LIBS += /usr/local/lib/libopencv_highgui.so  \
        /usr/local/lib/libopencv_highgui.so.3.0  \
        /usr/local/lib/libopencv_core.so     \
        /usr/local/lib/libopencv_imgproc.so  \
        /usr/local/lib/libopencv_imgcodecs.so \
        /usr/local/lib/libopencv_video.so \
        /usr/local/lib/libopencv_shape.so \
        /usr/local/lib/libopencv_videoio.so \
        /usr/local/lib/libopencv_core.so.3.0 \
        /usr/local/lib/libopencv_objdetect.so \
        /usr/local/lib/libopencv_face.so \
        /usr/local/lib/libopencv_face.so.3.0 \


HEADERS += \
    Utils.hpp \
    detectFace.hpp \
    Recognition.hpp \
    preprocessface.hpp \
    define.h


