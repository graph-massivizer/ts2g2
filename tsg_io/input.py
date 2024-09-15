import pandas as pd
import xml.etree.ElementTree as ET
import core.model as sg

class CsvRead:
    """Extracts data from a csv file.""" 
    def __init__(self):
        pass

    def from_csv(self):
        pass

class CsvFile(CsvRead):
    """
    Returns proccessed data from csv file sorted by "Date".
    
    **Attributes:**

    - `path`: path to csv file with data
    - `column`: which column of data do we extract
    """
    def __init__(self, path, y_column):
        self.path = path
        self.y_column = y_column
    
    def from_csv(self):
        """
        Extracts the data using set attributes.
        """
        timeseries = pd.read_csv(self.path)
        timeseries["Date"] = pd.to_datetime(timeseries["Date"])
        timeseries.set_index("Date", inplace=True)
        timeseries = timeseries[self.y_column]
        return timeseries

class XmlRead:
    """Extracts data from an xml file."""
    def __init__(self):
        pass

    def from_xml(self):
        pass

class FundamentalsReportFinancialStatements(XmlRead):
    """
    Extracting data from an xml file.
    Further explanation on file format can be found on:
    https://docs-2-0--quantrocket.netlify.app/data/reuters/
    
    **Attributes:**

    - `path`: path to csv file with data
    - `item`: which item are we observing
    - `season`: are we observing values annually or interim
    
    """
    def __init__(self, path, item, season = "Annual"):
        self.path = path
        self.item = item
        self.season = season
    
    def from_xml(self):
        """
        Extracts the data using set attributes.
        """
        tree = ET.parse(self.path)
        root = tree.getroot()

        financial_statements = root.find('FinancialStatements')
        COAMap = financial_statements.find('COAMap')
        
        periods = None
        if self.season.lower() == "annual":
            periods = financial_statements.find("AnnualPeriods")
        else:
            periods = financial_statements.find('InterimPeriods')

        elements = periods.findall(f".//lineItem[@coaCode = '{self.item}']")
        column = []

        for element in elements:
            column.append(float(element.text))
        
        return column