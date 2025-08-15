from vparam.HDR_parser.HDRParser import HDRParser


# A wrapper object representing the HDR data in the system
# It is created after HDR file is parsed by HDR parser

class FootageImgAtts:
    def __init__(self, wwl_units=None, wwl_vector=None, fwhm_vector=None, reflectance_scale=10000, map_info=None,
                 interleave=None, heigh=0, width=0, rgb=(0, 0, 0)):
        self.wwl_units = wwl_units  # default
        self.wwl_vector = [] if wwl_vector is None else wwl_vector
        self.fwhm_vector = [] if fwhm_vector is None else fwhm_vector
        self.reflectance_scale = reflectance_scale
        self.interleave = interleave
        self.rgb = list(rgb)
        self.height = heigh
        self.width = width
        self.wwl_reader_active = False
        self.fwhm_reader_active = False
        self.map_info = {'coord_system': '', 'ref_pixel_indexes': [0, 0], 'ref_pixel_coords': [0, 0],
                         'xy_px_size': [1, 1], 'zone': 33, 'north': True, 'idk_yet': '',
                         'units': 'meters'} if map_info is None else map_info


# Used for parsing of metadata from web request
def parse_from_json(json):
    # S2
    if json["sentinel"]:
        extracted_data = FootageImgAtts(wwl_units="nm",
                                        wwl_vector=[443.0, 490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0,
                                                    945.0, 1375.0, 1610.0, 2190.0],
                                        fwhm_vector=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], reflectance_scale=1)
    else:
        extracted_data = FootageImgAtts(wwl_units=json["picked_units"], wwl_vector=json["wavelengths"],
                                        fwhm_vector=json["fwhms"], reflectance_scale=json["reflectance_scale"])
    return extracted_data


def load_data(filename):
    # Create parser obj
    parser = HDRParser(filename)
    # Parse the file
    parser.parse_envi_file()

    hdr_data = parser.tokens

    return FootageImgAtts(wwl_units=hdr_data[""],
                          wwl_vector=hdr_data["wavelength"],
                          fwhm_vector=hdr_data["fwhm"],
                          reflectance_scale=hdr_data["reflectance scale factor"],
                          interleave=hdr_data["interleave"],
                          map_info=parser.extract_map_info(),
                          heigh=hdr_data["lines"],
                          width=hdr_data["samples"]
                          )
