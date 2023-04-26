import pandas as pd
import geopandas
from sklearn.preprocessing import MinMaxScaler


def percentage_bush_2004(states='all'):
    df = pd.read_csv('./data/countypres_2000-2020.csv')
    df_2004 = df.loc[(df.year==2004) & (df.party=='REPUBLICAN')]
    if states != 'all':
        df_2004 = df_2004.loc[df_2004.state.isin(states)]
    df_2004['bush_votes_perc'] = df_2004['candidatevotes'] / df_2004['totalvotes']
    df_2004 = df_2004.drop(df_2004[df_2004.bush_votes_perc.isnull()].index)
    df_2004['county_fips'] = df_2004['county_fips'].apply(int)

    contiguity = geopandas.read_file('../data/ncovr/NAT.shp')
    contiguity['FIPS'] = contiguity['FIPS'].apply(int)

    gdf = contiguity.merge(df_2004, left_on='FIPS', right_on='county_fips')
    gdf['neighbors'] = [[]]*len(gdf)
    for index, county in gdf.iterrows():
        # get 'not disjoint' countries
        neighbors = gdf[~gdf.geometry.disjoint(county.geometry)].county_fips.tolist()
        neighbors.remove(county.county_fips)

        # add names of neighbors as NEIGHBORS value
        gdf.at[index, 'neighbors'] = neighbors
    gdf = gdf.set_index('county_fips',drop=True)#[['county_fips', 'county_name', 'state', 'state_po', 'bush_votes_perc','candidatevotes','totalvotes','neighbors','geometry']].set_index('county_fips',drop=True)
    return gdf

def ecodemoEurope(indicators,year,level):
    df = None
    for ind in indicators:
        df_ind = pd.read_csv('./data/EcoDemoEurope/'+ind+'.csv')[['geo','TIME_PERIOD','OBS_VALUE']]
        df_ind.rename(columns={'OBS_VALUE':ind},inplace=True)
        if df is None:
            df = df_ind
        else:
            df = pd.merge(left=df,right=df_ind,on=['geo','TIME_PERIOD'],how='inner')
    df = df[df['TIME_PERIOD']==year]
    df = df[df['geo'].apply(lambda row: row[:2] != 'CY' and row[:2] != 'IE' and row[:2] != 'MT')]
    #df = df[df['geo'].apply(lambda row: len(row)==2+block_size)]
    gdf = geopandas.read_file('../data/EcoDemoEurope/NUTS_RG_20M_2021_3035.shp')
    gdf = gdf[gdf['LEVL_CODE']==level]
    gdf = gdf.merge(df,left_on='NUTS_ID',right_on='geo')
    gdf['neighbors'] = [[] for _ in range(len(gdf))]
    for index, row in gdf.iterrows():
        # get 'not disjoint' countries
        neighbors = gdf[~gdf.geometry.disjoint(row.geometry)].NUTS_ID.tolist()
        neighbors.remove(row.NUTS_ID)

        # add names of neighbors as NEIGHBORS value
        gdf.at[index, 'neighbors'] = neighbors
    scaler = MinMaxScaler()
    gdf[indicators] = scaler.fit_transform(gdf[indicators])
    gdf = gdf.set_index('NUTS_ID',drop=True)
    visited = []
    to_expand = [gdf.index[0]]
    while to_expand:
        next = to_expand[0]
        if next not in visited:
            visited.append(next)
            to_expand += gdf['neighbors'][next]
        to_expand = to_expand[1:]
    gdf = gdf.loc[visited]
    return gdf

def ecoregions(indicators):
    gdf = geopandas.read_file('./data/Ecoregions/Ecoregions.shp')
    to_remove = [12, 15, 18, 19, 25, 35, 47, 56, 61, 160, 165, 174, 177, 184]
    gdf = gdf.loc[[i for i in range(len(gdf)) if i not in to_remove]]
    gdf = gdf.reset_index()
    gdf['neighbors'] = [[] for _ in range(len(gdf))]
    for index, row in gdf.iterrows():
        # get 'not disjoint' countries
        neighbors = gdf[~gdf.geometry.disjoint(row.geometry)].index.tolist()
        neighbors.remove(index)

        # add names of neighbors as NEIGHBORS value
        gdf.at[index, 'neighbors'] = neighbors
    scaler = MinMaxScaler()
    gdf[indicators] = scaler.fit_transform(gdf[indicators])
    return gdf

def education_BE():
    df = pd.read_excel('./data/EcoBelgium/BE_Education_2017.xlsx')
    gdf = geopandas.read_file('../data/EcoBelgium/communes-belges-2019.shp')
    gdf[['EDU_LOW_r','EDU_MID_r','EDU_HIGH_r']] = None
    gdf['niscode'] = pd.to_numeric(gdf['niscode'])
    gdf = gdf.set_index('niscode')
    to_drop = []
    for ind in gdf.index:
        commune = df.loc[df['CD_MUNTY_REFNIS'] == ind]
        if len(commune) < 4:
            to_drop.append(ind)
        else:
            tot_pop = commune.loc[commune['CD_ISCED_CL']=='TOT']['MS_POPULATION_25'].array[0]
            low_pop = commune.loc[commune['CD_ISCED_CL'] == 'LOW']['MS_POPULATION_25'].array[0]
            mid_pop = commune.loc[commune['CD_ISCED_CL'] == 'MIDDLE']['MS_POPULATION_25'].array[0]
            high_pop = commune.loc[commune['CD_ISCED_CL'] == 'HIGH']['MS_POPULATION_25'].array[0]
            gdf.loc[ind,['EDU_LOW_r', 'EDU_MID_r', 'EDU_HIGH_r']] = [low_pop/tot_pop,mid_pop/tot_pop,high_pop/tot_pop]
    gdf['neighbors'] = [[] for _ in range(len(gdf))]
    gdf = gdf.drop(to_drop)
    for index, row in gdf.iterrows():
        # get 'not disjoint' countries
        neighbors = gdf[~gdf.geometry.disjoint(row.geometry)].index.tolist()
        neighbors.remove(index)

        # add names of neighbors as NEIGHBORS value
        gdf.at[index, 'neighbors'] = neighbors
    return gdf

def generate_dist_matrix(df, cols, dist_deg=2):
        Full_order_edges = {}
        indexes = df.index
        dist_matrix = pd.DataFrame(0, index=indexes, columns=indexes)
        for i in range(len(indexes) - 1):
            index1 = indexes[i]
            for j in range(i + 1, len(indexes)):
                index2 = indexes[j]
                dist = 0
                for col in cols:
                    dist += (df.loc[index1, col] - df.loc[index2, col]) ** dist_deg
                dist_matrix.loc[index1, index2] = dist
                dist_matrix.loc[index2, index1] = dist
                if index1 < index2:
                    Full_order_edges[(index1, index2)] = dist
                else:
                    Full_order_edges[(index2, index1)] = dist
        return Full_order_edges, dist_matrix

def generate_cont_matrix(df):
    contiguity_matrix = pd.DataFrame(0, index=df.index, columns=df.index)
    for index, row in df.iterrows():
        for n in row.neighbors:
            contiguity_matrix.loc[index, n] = 1
    return contiguity_matrix