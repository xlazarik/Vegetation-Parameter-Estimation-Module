from rest_framework import serializers
from vparam.models import Lut


class LutSerializer(serializers.ModelSerializer):
    class Meta:
        many = True
        model = Lut
        fields = ('lut_id', 'lut_name', 'created_on', 'crop_type', 'delimeter', 'has_header', 'units',
                  'simulation_step', 'approximated_parameters', 'filtering_parameters', 'resampled_to_sentinel',
                  'fragmented', 'no_of_lines', 'author_information')
