import re


class CharacterFeatures():
    def __init__(self):
        pass

    def count_character(self, text, char) -> float:
        if text:
            return text.count(char)
        else:
            return 0

    def character_flooding(self, text):
        ...

    def total_number_of_chars(self, text)  -> float:
        return sum(1 for c in text if not c.isspace())

    def ratio_of_upper_case_and_lower_case_chars(self, text) -> tuple[float, float]:
        """
        Calulate the ratio between upper and lower characters in a text.
        Divide the count by all letters, which excludes.
        Denominator are only alphanumeric chars!
        :param text:
        :return:
        """
        letters = [c for c in text if c.isalpha()]
        if len(letters) == 0:
            return 0,0
        is_upper_ratio = sum((1 for c in letters if c.isupper()))/len(letters)
        is_lower_ratio = sum((1 for c in letters if c.islower()))/len(letters)
        return is_upper_ratio, is_lower_ratio

    def ratio_of_periods(self, text) -> float:
        """
        Calulates periods per token.
        :param text:
        :return:
        """
        if len(text) == 0:
            return 0
        tokens = [t for t in text if not t.isspace()]
        periods = sum(1 for t in text if t==".")
        #print(tokens)
        #print(periods)
        return periods/len(tokens)

    def ratio_of_commas(self, text) -> float:
        """
        Calulates commas per token.
        :param text:
        :return:
        """
        if len(text) == 0:
            return 0
        tokens = [t for t in text if not t.isspace()]
        periods = sum(1 for t in text if t==",")
        #print(tokens)
        #print(periods)
        return periods/len(tokens)

    def ratio_of_semicolons(self, text) -> float:
        """
        Calulates periods per token.
        :param text:
        :return:
        """
        if len(text) == 0:
            return 0
        tokens = [t for t in text if not t.isspace()]
        periods = sum(1 for t in text if t==";")
        #print(tokens)
        #print(periods)
        return periods/len(tokens)


    def ratio_of_colons(self, text) -> float:
        """
        Calulates periods per token.
        :param text:
        :return:
        """
        if len(text) == 0:
            return 0
        tokens = [t for t in text if not t.isspace()]
        periods = sum(1 for t in text if t==":")
        return periods/len(tokens)

    def ratio_of_exclamation(self, text) -> float:
        """
        Calulates periods per token.
        :param text:
        :return:
        """
        if len(text) == 0:
            return 0
        chars = [c for c in text if not c.isspace()]
        periods = sum(1 for t in text if t=="!")
        return periods/len(chars)

    def ratio_of_parentheses(self, text):
        """
        Calulates periods per token.
        :param text:
        :return:
        """
        if len(text) == 0:
            return 0
        chars = [c for c in text if not c.isspace()]
        return sum(c in "()" for c in chars) / len(chars)


    def ratio_of_numbers(self,text):
        if len(text) == 0:
            return 0
        tokens = [t for t in text if not t.isspace()]
        numbers = sum(1 for t in text if t.isdigit()) # not numeric, because we do not want to include  Unicode numeric value property, e.g U+2155
        return numbers/len(tokens)

    def ratio_of_hyphens(self,text):
        if len(text) == 0:
            return 0
        tokens = [t for t in text if not t.isspace()]
        numbers = sum(1 for t in text if t == "-") # not numeric, because we do not want to include  Unicode numeric value property, e.g U+2155
        return numbers/len(tokens)

    def ratio_of_quotation_marks(self, text):
        """
        Calulates quotation marks. Single quotes are tricky, since they can also be used in don't or you're.
        So this case need to be handled: Single quotes are only counted if they are not between chars
        :param text:
        :return:
        """
        if len(text) == 0:
            return 0
        chars = [c for c in text if not c.isspace()]
        QUOTES = {'"', '“', '”', '‘', '’'}
        quotes_count = 0
        for i, c in enumerate(text):
            if c in QUOTES:
                quotes_count += 1
            if c == "'": # handle special case of single quotation mark, get the environment of the quotation mark
                previous_char = text[i-1] if i > 0 else " "
                next_char = text[i+1] if i < len(text)-1 else " "
                # check the environment
                if not (previous_char.isalpha() and next_char.isalpha()):
                    quotes_count+=1
        return quotes_count/len(chars)

