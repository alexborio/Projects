from sqlalchemy import create_engine
from sqlalchemy.schema import CreateTable
from sqlalchemy.orm import sessionmaker
import numpy as np
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float

Base = declarative_base()


class Data(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True)
    num = Column(Integer)
    txt = Column(String)
    flt = Column(Float)

    # optional __repr__ method
    def __repr__(self):
        return "<Data(num='%i', txt='%s', flt='%f')>" % (
            self.num, self.txt, self.flt)


engine = create_engine('postgresql+psycopg2://user:user@localhost/db')
Session = sessionmaker(bind=engine)
session = Session()


names=('id', 'num', 'txt', 'flt')

data = ((0, np.random.randint(0, 100000), 'kfjlsdf', np.random.randn()),)

for i in range(100):
    data = data + ((i + 1, np.random.randint(0, 1000), 'kfjlsdf', np.random.randn()),)

data = (names, data)
data_df = pd.DataFrame(list(data[1]), columns=list(data[0]))

Base.metadata.create_all(engine) # create schema

session.bulk_insert_mappings(Data, data_df.to_dict(orient='records'))
session.commit()
session.close()