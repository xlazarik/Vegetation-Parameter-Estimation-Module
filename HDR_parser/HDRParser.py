import json
import os


# Parsing of ENVI HDR files
# File can be parsed by creating HDRParser object and passing it the filepath
# Parsing starts by calling parse_envi_file()
class HDRParser:
    def __init__(self, filename, delimeter=","):
        self.file = open(filename, "r")
        self.delimeter = delimeter
        self.tokens = dict()
        self.next_token = ""
        self.token_value = ""
        self.reading_value = False
        self.array_brackets_stack = []

    def file_peek(self):
        next_character = self.file.read(1)
        self.file.seek(self.file.tell() - 1)
        return next_character

    def produce_token_keyword(self, character):
        if self.next_token == "ENVI":
            self.next_token = ""
            return
        # Newline ignored
        if character == "\r\n" or character == "\n":
            return
        # Whitespaces
        if character == " ":
            # Shrink multiple whitespaces to one (does not change the meaning)
            if self.file_peek() == " ":
                return
            # New token, its value follows
            if self.file_peek() == "=":
                # Array closing bracket missing
                if len(self.array_brackets_stack) != 0:
                    raise Exception()
                # self.tokens["next_token"] = None
                self.reading_value = True
                self.file.read(1)
                return
        self.next_token += character

    def produce_token_value(self, character):
        # Newline ignored
        if character == "\r\n" or character == "\n":
            # Scalar token end
            if len(self.array_brackets_stack) == 0:
                self.try_parse_token_value()
                self.tokens[self.next_token] = self.token_value
                self.reading_value = False
                self.next_token = ""
                self.token_value = ""
            return

        if character == " ":
            # Shrink multiple whitespaces to one (does not change the meaning)
            if self.file_peek() == " ":
                return
            # Did not start reading the value, padding at the beginning, skipping
            if self.token_value == "":
                return
        # Array starting bracket
        if character == "{":
            # Array closing bracket missing
            if len(self.array_brackets_stack) != 0:
                raise Exception()
            self.tokens[self.next_token] = []
            self.array_brackets_stack.append("{")
            return

        # Array closed
        if character == "}":
            if self.array_brackets_stack[-1] != "{":
                raise Exception()
            self.array_brackets_stack.pop()
            self.try_parse_token_value()
            self.tokens[self.next_token].append(self.token_value)
            self.reading_value = False
            self.next_token = ""
            self.token_value = ""
            return
        if character == "[":
            self.array_brackets_stack.append("[")
        if character == "]":
            if self.array_brackets_stack[-1] != "[":
                raise Exception()
            self.array_brackets_stack.pop()

        if character == self.delimeter:
            # Nested array, part of content, does not imply new array token value array item
            if self.array_brackets_stack and self.array_brackets_stack[-1] == "[":
                return
            self.try_parse_token_value()
            self.tokens[self.next_token].append(self.token_value)
            self.token_value = ""
            return

        self.token_value += character

    # "Main" fnc
    def parse_envi_file(self):
        try:
            while True:
                character = self.file.read(1)
                if not character:
                    print("End of file")
                    break
                if self.reading_value:
                    self.produce_token_value(character)
                else:
                    self.produce_token_keyword(character)
        except Exception as e:
            print(e)
        finally:
            self.file.close()

    def try_parse_token_value(self):
        try:
            self.token_value = int(self.token_value)
            return
        # Exc only occurs when the value is not numeric -> its OK
        except Exception:
            pass
        try:
            self.token_value = float(self.token_value)
            return
        except Exception:
            pass

    def convert_units(self, from_un="nm", to_un="um"):
        factor = 0
        if from_un == "nm" and to_un == "um":
            factor = 1000
        if from_un == "um" and to_un == "nm":
            factor = 0.001

        if "wavelength" in self.tokens:
            for i in range(len(self.tokens["wavelength"])):
                self.tokens["wavelength"][i] = self.tokens["wavelength"][i] * factor
                if "fwhm" in self.tokens:
                    self.tokens["fwhm"][i] = self.tokens["fwhm"][i] * factor

    # Custom function for the purpose of this use case
    def extract_map_info(self):
        map_info = dict()
        if "map info" in self.tokens:
            info = self.tokens["map info"]
            map_info['coord_system'] = info[0]
            map_info['ref_pixel_indexes'] = [info[1], info[2]]
            map_info['ref_pixel_coords'] = [info[3], info[4]]
            map_info['xy_px_size'] = [info[5], info[6]]
            map_info['zone'] = info[7]
            map_info['north'] = info[8].lower().find("north") != -1
            map_info['idk_yet'] = info[9]
            map_info['units'] = info[10][info[10].find("=") + 1:]
        return map_info

    def save_metadata_json(self, filepath):
        filename = os.path.basename(filepath).split(".")[0]
        path_root = os.path.dirname(filepath)
        with open(path_root + os.path.sep + filename + "_metadata.json", "w") as filehandle:
            json.dump(self.tokens, filehandle)

    def serialize_as_json(self):
        return json.dumps(self.tokens)
