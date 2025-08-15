from django.db import models

from mosveg.models import User


class LutFragment(models.Model):
    fragment_id = models.AutoField(primary_key=True)
    base_lut = models.ForeignKey(Lut, on_delete=models.CASCADE)
    fragment_name = models.CharField(max_length=50)
    fragment_dir = models.CharField(max_length=250)
    low_wavelength_bound = models.FloatField()
    high_wavelength_bound = models.FloatField()

    def __str__(self):
        return f"Fragment from {self.base_lut.lut_name} within: " \
               f"{self.low_wavelength_bound}-{self.high_wavelength_bound} {self.base_lut.units}"


class ResampledLutCache(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    resampled_lut_id = models.AutoField(primary_key=True)
    base_lut = models.ForeignKey(Lut, on_delete=models.CASCADE)
    resampling_dir = models.CharField(max_length=450)
    resampled_to_wavelengths = models.CharField(max_length=2000)
    created_on = models.DateTimeField(auto_now_add=True)
    sentinel = models.BooleanField(default=False)

    def __str__(self):
        return f"Resampled Lut cache for Lut  {self.base_lut.lut_name} for wavelengths: " \
               f"{self.resampled_to_wavelengths}"


# TODO: SRF can have own object -> especially when there will be more supported satellites
class Lut(models.Model):
    lut_id = models.AutoField(primary_key=True)
    created_on = models.DateTimeField(auto_now_add=True)
    lut_filename = models.CharField(max_length=250)
    param_filename = models.CharField(max_length=250)
    lut_name = models.CharField(max_length=30)
    lut_dir = models.CharField(max_length=250)
    resampled_to_sentinel = models.BooleanField(default=False)
    fragmented = models.BooleanField(default=False)
    no_of_lines = models.IntegerField()
    crop_type = models.CharField(max_length=15)
    delimeter = models.CharField(max_length=3)
    has_header = models.BooleanField(default=True)
    units = models.CharField(max_length=12)
    simulation_step = models.IntegerField(default=1)
    approximated_parameters = models.CharField(max_length=400)
    filtering_parameters = models.CharField(max_length=400)
    original_parameters = models.CharField(max_length=400)
    lut_scaling_factor = models.FloatField(default=1)
    lut_wavelength_interval = models.CharField(max_length=150)
    srf_filename = models.CharField(max_length=250)
    srf_wavelengths = models.CharField(max_length=150)
    srf_file_wavelength_interval = models.CharField(max_length=60)
    srf_header = models.BooleanField(default=True)
    srf_indexed = models.BooleanField(default=True)
    author_information = models.TextField(default="")

    def __str__(self):
        return self.lut_name
