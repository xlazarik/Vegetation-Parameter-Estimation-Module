from vparam.views import *
from django.urls import path

urlpatterns = [
    path('hdr/', HdrView.as_view(), name="hdr"),
    path('filetype/convert/', FiletypeConvert.as_view(), name="converting"),
    path('computation/submit/', ComputationController.as_view(), name="submit_computation"),
    path('lut/mount/', LutController.as_view(), name="mount_lut"),
    path('lut/delete/', LutController.as_view(), name="delete_lut"),
    path('lut/previews/', LutController.as_view(), name="fetch_lut_preview"),
    path('lut/databases', ComputationController.as_view(), name="fetch_lut_databases"),
    path('fileutil/countlines/', VparamUtil.as_view(), name="count_lines"),
    path('fileutil/lut_parambounds/', VparamUtil.as_view(), name="lut_parambounds"),
    path('computation/previews/', PreviewImage.as_view(), name="preview_image"),
]
