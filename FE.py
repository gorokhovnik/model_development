from model_development import *


class FE:
    def __init__(self,
                 file,
                 tables_schema='',
                 tables_prefix='bki_',
                 table_for_each_variable=True,
                 include_list=None):
        self.__global_flt_table = pd.read_excel(file, 'global_flt').query('include == 1').fillna('')
        self.__flt_table = pd.read_excel(file, 'flt').query('include == 1').fillna('')
        self.__vars_table = pd.read_excel(file, 'vars').query('include == 1').fillna('')
        self.__overview_vars_table = pd.read_excel(file, 'overview_vars').query('include == 1').fillna('')
        self.__ratio_vars_table = pd.read_excel(file, 'ratio_vars').query('include == 1').fillna('')
        self.__sources_table = pd.read_excel(file, 'sources').query('include == 1').fillna('')
        self.__processes_table = pd.read_excel(file, 'processes').query('include == 1').fillna('')

        self.__appl_info = self.__sources_table[self.__sources_table['type'] == 'Заявка'].to_dict('records')[0]
        self.__appl_info['schema.table'] = self.__appl_info['schema'] + '.' + self.__appl_info['table']
        self.__over_info = self.__sources_table[self.__sources_table['type'] == 'Общий'].to_dict('records')[0]
        self.__over_info['schema.table'] = self.__over_info['schema'] + '.' + self.__over_info['table']
        self.__loans_info = self.__sources_table[self.__sources_table['type'] == 'Кредиты'].to_dict('records')[0]
        self.__loans_info['schema.table'] = self.__loans_info['schema'] + '.' + self.__loans_info['table']
        self.__pay_info = self.__sources_table[self.__sources_table['type'] == 'Платежи'].to_dict('records')[0]
        self.__pay_info['schema.table'] = self.__pay_info['schema'] + '.' + self.__pay_info['table']

        self.__processes = self.__processes_table.set_index('process')['suffix'].to_dict()

        self.__pay_names = self.__vars_table[(self.__vars_table['type'] == 'Платежи')]['name'].tolist()
        self.__pay_loan_names = self.__vars_table[(self.__vars_table['type'] == 'Платежи по кредитам')]['name'].tolist()
        self.__loan_names = self.__vars_table[(self.__vars_table['type'] == 'Кредиты')]['name'].tolist()
        self.__var_names = self.__pay_loan_names + self.__loan_names
        self.__over_names = self.__overview_vars_table['name'].tolist()

        self.__ratio_names = list(set(
            [self.__ratio_name(row['num_name']) + '_TO_' + self.__ratio_name(row['den_name']) + '_RATIO'
             for i, row in self.__ratio_vars_table.iterrows()]))

        self.__names = self.__var_names + self.__over_names + self.__ratio_names

        self.__schema = tables_schema
        self.__prefix = tables_prefix
        self.__schema_prefix = self.__schema + '.' + self.__prefix
        self.__table_for_each_variable = table_for_each_variable
        self.__include_list = include_list if include_list != [] else None

        if self.__table_for_each_variable:
            self.__tables = self.__var_names + ['overview'] + self.__ratio_names
        else:
            self.__tables = ['payments_agg', 'loans_agg', 'overview', 'ratio']

        self.__query = {p: {} for p in self.__processes}
        self.__suf = '#suffix#'

    def __ratio_name(self, name):
        return name if name in self.__var_names else 'Overview'

    def __form_flt_vector(self, comb):
        combinations = re.sub('\+', '|', re.sub('[,;]', ',', re.sub(' +', '', comb))).split(',')
        combination_list = []
        code_dict = {}

        for combination in combinations:
            filter_comb = self.__flt_table[
                self.__flt_table['filter_id'].str.contains(combination)][['cat', 'filter_id', 'code']]
            code_dict.update(filter_comb.set_index('filter_id')['code'].to_dict())
            filter_ids = filter_comb.pivot(columns='cat', values='filter_id').values.T
            filter_ids = [[filter_ids[i][j] for j in range(len(filter_ids[i])) if filter_ids[i][j] == filter_ids[i][j]]
                          for i in range(len(filter_ids))]
            filter_ids = list(product(*filter_ids))
            for f in filter_ids:
                combination_list += [tuple(sorted(f, key=lambda s: s.upper()))]

        suffix, code = [], []
        for filters in combination_list:
            suffix += ['_'.join(filters)]
            code += [' AND '.join([code_dict[flt] for flt in filters])]
        return suffix, code

    def __form_vars(self, var):
        varsuffix, varcode = self.__form_flt_vector(var['comb'])
        vartype = var['type']
        varfuns = var['funs']
        varname = var['name']
        query = []
        for suffix, code in zip(varsuffix, varcode):
            funs = re.sub('flt', code, varfuns)
            if vartype == 'Платежи':
                pay_var = varname + '_' + suffix
                if self.__include_list is None or any(pay_var in include_var for include_var in self.__include_list):
                    self.__pay_vars[varname] += [pay_var]
                    query += [funs + ' AS ' + pay_var]
            elif vartype == 'Платежи по кредитам':
                for base_pay_var in self.__pay_vars:
                    if base_pay_var in varname:
                        for pay_var in self.__pay_vars[base_pay_var]:
                            if base_pay_var in varname:
                                pay_loan_var = re.sub(base_pay_var, pay_var, varname) + '_' + suffix
                                if self.__include_list is None or any(pay_loan_var in include_var
                                                                      for include_var in self.__include_list):
                                    self.__pay_loan_vars[varname] += [pay_loan_var]
                                    self.__var_vars[varname] += [pay_loan_var]
                                    query += [re.sub(base_pay_var, pay_var, funs) + ' AS ' + pay_loan_var]
                        break
            elif vartype == 'Кредиты':
                loan_var = varname + '_' + suffix
                if self.__include_list is None or any(loan_var in include_var for include_var in self.__include_list):
                    self.__loan_vars[varname] += [loan_var]
                    self.__var_vars[varname] += [loan_var]
                    query += [funs + ' AS ' + loan_var]
        return ',\n'.join(query)

    def __form_overview_vars(self):
        funs = self.__overview_vars_table['funs'].tolist()
        name = self.__overview_vars_table['name'].tolist()
        query = []
        for f, n in zip(funs, name):
            if self.__include_list is None or n in self.__include_list:
                self.__over_vars += [n]
                query += [f + ' AS ' + n]
        return ',\n'.join(query)

    def __form_ratio_vars(self, var):
        query = []
        varnumname = var['num_name']
        vardenname = var['den_name']
        varname = self.__ratio_name(varnumname) + '_TO_' + self.__ratio_name(vardenname) + '_RATIO'
        num_vars = []
        den_vars = []
        if varnumname not in self.__var_names:
            num_vars = grepl(varnumname, self.__over_vars)
        else:
            varnumcomb = re.sub('\+', '.*', re.sub('[,;]', ',', re.sub(' +', '', var['num_comb']))).split(',')
            for num_comb in varnumcomb:
                num_vars = grepl(num_comb, self.__var_vars[varnumname])
        if vardenname not in self.__var_names:
            den_vars = grepl(vardenname, self.__over_vars)
        else:
            vardencomb = re.sub('\+', '.*', re.sub('[,;]', ',', re.sub(' +', '', var['den_comb']))).split(',')
            for den_comb in vardencomb:
                den_vars = grepl(den_comb, self.__var_vars[vardenname])

        def convert(k):
            return re.sub('%', '', re.sub('([^0-9]|^)([0-9])([^0-9]|$)', '\\1%0\\2\\3', re.sub('00', 'AA', k)))

        rat_vars = [p for p in product(num_vars, den_vars) if p[0] != p[1]]
        if varnumname == vardenname:
            rat_vars = list(set([tuple(sorted(r, key=convert)) for r in rat_vars]))
        rat_vars = sorted(rat_vars, key=lambda k: convert(k[0] + '_TO_' + k[1]))

        for r in rat_vars:
            name = r[0] + '_TO_' + r[1] + '_RATIO'
            if self.__include_list is None or name in self.__include_list:
                self.__ratio_vars[varname] += [name]
                query += [r[0] + ' * 1.0 / NULLIF(' + r[1] + ', 0) AS ' + name]

        return ',\n'.join(query)

    def __form_global_flt(self, comb):
        combination = re.sub('\+', '|', re.sub(' +', '', comb))
        filter_comb = self.__global_flt_table[self.__global_flt_table['filter_id'].str.contains(combination)]['code']
        return ' AND '.join(filter_comb.to_list())

    def __make_query(self,
                     select_txt,
                     into_txt,
                     from_txt,
                     where_txt=None,
                     groupby_txt=None,
                     drop=False):
        query = ''
        if drop:
            query += 'DROP TABLE ' + into_txt + '\n'
        query += 'SELECT TOP 0 ' + select_txt + '\n'
        query += 'INTO ' + into_txt + '\n'
        query += 'FROM ' + from_txt + '\n'
        if groupby_txt is not None:
            query += 'GROUP BY ' + groupby_txt + '\n'

        query += '\n\n\n'

        query += 'INSERT INTO ' + into_txt + '\n'
        query += 'SELECT ' + select_txt + '\n'
        query += 'FROM ' + from_txt
        if where_txt is not None:
            query += '\n' + 'WHERE ' + where_txt
        if groupby_txt is not None:
            query += '\n' + 'GROUP BY ' + groupby_txt

        if self.__use_compression:
            query += '\n' + 'ALTER TABLE ' + into_txt + '\n'
            query += 'REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)'

        return query

    def __form_loans_and_payments_script(self):
        var_queries = []
        for idx, var in self.__vars_table[self.__vars_table['type'] == 'Платежи'].iterrows():
            var_queries += [self.__form_vars(var.to_dict())]

        select_txt = self.__pay_info['group_by'] + ',\n' + ',\n'.join(var_queries)
        into_txt = '#payments'
        from_txt = self.__pay_info['schema.table'] + self.__suf
        groupby_txt = self.__pay_info['group_by']
        query = self.__make_query(select_txt=select_txt,
                                  into_txt=into_txt,
                                  from_txt=from_txt,
                                  groupby_txt=groupby_txt,
                                  drop=True) + '\n\n\n\n'

        select_txt = 'loans.*,\n' + ',\n'.join(unpack_dict(self.__pay_vars))
        into_txt = self.__schema_prefix + 'loans_and_payments' + self.__suf
        from_txt = self.__loans_info['schema.table'] + self.__suf + ' AS loans\n'
        from_txt += 'LEFT JOIN #payments AS payments\n'
        from_txt += 'ON loans.' + self.__loans_info['id'] + ' = payments.' + self.__pay_info['group_by']
        where_txt = re.sub(self.__pay_info['group_by'], 'loans.' + self.__pay_info['group_by'],
                           self.__form_global_flt(self.__loans_info['comb']))

        query += self.__make_query(select_txt=select_txt,
                                   into_txt=into_txt,
                                   from_txt=from_txt,
                                   where_txt=where_txt,
                                   drop=self.__drop_table_before_creating)

        return {'loans_and_payments': query}

    def __form_vars_script(self):
        query_dict = {}
        if self.__table_for_each_variable:
            for idx, var in self.__vars_table[self.__vars_table['type'] != 'Платежи'].iterrows():
                select_txt = self.__loans_info['group_by'] + ',\n' + self.__form_vars(var.to_dict())
                into_txt = self.__schema_prefix + var['name'] + self.__suf
                from_txt = self.__schema_prefix + 'loans_and_payments' + self.__suf
                groupby_txt = self.__loans_info['group_by']

                query = self.__make_query(select_txt=select_txt,
                                          into_txt=into_txt,
                                          from_txt=from_txt,
                                          groupby_txt=groupby_txt,
                                          drop=self.__drop_table_before_creating)

                if var['type'] == 'Кредиты':
                    query_dict['loans_agg|' + var['name']] = query
                elif var['type'] == 'Платежи по кредитам':
                    query_dict['payments_agg|' + var['name']] = query
        else:
            var_queries = []
            for idx, var in self.__vars_table[self.__vars_table['type'] == 'Кредиты'].iterrows():
                var_queries += [self.__form_vars(var.to_dict())]

            select_txt = self.__loans_info['group_by'] + ',\n' + ',\n'.join(var_queries)
            into_txt = self.__schema_prefix + 'loans_agg' + self.__suf
            from_txt = self.__schema_prefix + 'loans_and_payments' + self.__suf
            groupby_txt = self.__loans_info['group_by']

            query_dict['loans_agg'] = self.__make_query(select_txt=select_txt,
                                                        into_txt=into_txt,
                                                        from_txt=from_txt,
                                                        groupby_txt=groupby_txt,
                                                        drop=self.__drop_table_before_creating)

            var_queries = []
            for idx, var in self.__vars_table[self.__vars_table['type'] == 'Платежи по кредитам'].iterrows():
                var_queries += [self.__form_vars(var.to_dict())]

            select_txt = self.__loans_info['group_by'] + ',\n' + ',\n'.join(var_queries)
            into_txt = self.__schema_prefix + 'payments_agg' + self.__suf

            query_dict['payments_agg'] = self.__make_query(select_txt=select_txt,
                                                           into_txt=into_txt,
                                                           from_txt=from_txt,
                                                           groupby_txt=groupby_txt,
                                                           drop=self.__drop_table_before_creating)

        return query_dict

    def __form_overview_vars_script(self):
        var_query = self.__form_overview_vars()

        select_txt = 'appl.' + self.__appl_info['id'] + ',\n' + var_query
        into_txt = self.__schema_prefix + 'overview' + self.__suf
        from_txt = self.__appl_info['schema.table'] + self.__suf + ' AS appl\n'
        from_txt += 'LEFT JOIN ' + self.__over_info['schema.table'] + self.__suf + ' AS ove\n'
        from_txt += 'ON appl.' + self.__appl_info['id'] + ' = ove.' + self.__over_info['id']
        where_txt = re.sub(self.__appl_info['id'], 'appl.' + self.__appl_info['id'],
                           self.__form_global_flt(self.__appl_info['comb']))
        where_txt += ' AND ' + re.sub(self.__over_info['id'], 'appl.' + self.__over_info['id'],
                                      self.__form_global_flt(self.__over_info['comb']))

        query = self.__make_query(select_txt=select_txt,
                                  into_txt=into_txt,
                                  from_txt=from_txt,
                                  where_txt=where_txt,
                                  drop=self.__drop_table_before_creating)

        return {'overview': query}

    def __form_ratio_vars_script(self):
        query_dict = {}
        var_queries = {var: [] for var in self.__ratio_names}
        for idx, var in self.__ratio_vars_table.iterrows():
            name = self.__ratio_name(var['num_name']) + '_TO_' + self.__ratio_name(var['den_name']) + '_RATIO'
            var_queries[name] += [self.__form_ratio_vars(var.to_dict())]
        if self.__table_for_each_variable:
            for name in self.__ratio_names:
                numname, denname = name[:-6].split('_TO_')
                if numname == denname and numname == 'Overview':
                    select_txt = self.__appl_info['id'] + ',\n' + ',\n'.join(var_queries[name])
                    from_txt = self.__schema_prefix + 'overview' + self.__suf
                elif numname == denname and numname != 'Overview':
                    select_txt = self.__loans_info['group_by'] + ',\n' + ',\n'.join(var_queries[name])
                    from_txt = self.__schema_prefix + numname + self.__suf
                elif numname == 'Overview':
                    select_txt = 'ove.' + self.__appl_info['id'] + ',\n' + ',\n'.join(var_queries[name])
                    from_txt = self.__schema_prefix + 'overview' + self.__suf + ' AS ove\n'
                    from_txt += 'LEFT JOIN ' + self.__schema_prefix + denname + self.__suf + ' AS ' + denname + '\n'
                    from_txt += 'ON ove.' + self.__appl_info['id']
                    from_txt += ' = ' + denname + '.' + self.__loans_info['group_by']
                elif denname == 'Overview':
                    select_txt = 'ove.' + self.__appl_info['id'] + ',\n' + ',\n'.join(var_queries[name])
                    from_txt = self.__schema_prefix + 'overview' + self.__suf + ' AS ove\n'
                    from_txt += 'LEFT JOIN ' + self.__schema_prefix + numname + self.__suf + ' AS ' + numname + '\n'
                    from_txt += 'ON ove.' + self.__appl_info['id']
                    from_txt += ' = ' + numname + '.' + self.__loans_info['group_by']
                else:
                    select_txt = numname + '.' + self.__loans_info['group_by'] + ',\n' + ',\n'.join(var_queries[name])
                    from_txt = self.__schema_prefix + numname + self.__suf + ' AS ' + numname + '\n'
                    from_txt += 'LEFT JOIN ' + self.__schema_prefix + denname + self.__suf + ' AS ' + denname + '\n'
                    from_txt += 'ON ' + numname + '.' + self.__loans_info['group_by']
                    from_txt += ' = ' + denname + '.' + self.__loans_info['group_by']
                into_txt = self.__schema_prefix + name + self.__suf

                query_dict['ratio|' + name] = self.__make_query(select_txt=select_txt,
                                                                into_txt=into_txt,
                                                                from_txt=from_txt,
                                                                drop=self.__drop_table_before_creating)
        else:
            select_txt = 'ove.' + self.__appl_info['id'] + ',\n' + ',\n'.join(unpack_dict(var_queries))
            into_txt = self.__schema_prefix + 'ratio' + self.__suf
            from_txt = self.__schema_prefix + 'overview' + self.__suf + ' AS ove' + '\n'
            from_txt += 'LEFT JOIN ' + self.__schema_prefix + 'loans_agg' + self.__suf + ' AS loans' + '\n'
            from_txt += 'ON ove.' + self.__appl_info['id'] + ' = loans.' + self.__loans_info['group_by'] + '\n'
            from_txt += 'LEFT JOIN ' + self.__schema_prefix + 'payments_agg' + self.__suf + ' AS pay' + '\n'
            from_txt += 'ON ove.' + self.__appl_info['id'] + ' = pay.' + self.__loans_info['group_by']

            query_dict['ratio'] = self.__make_query(select_txt=select_txt,
                                                    into_txt=into_txt,
                                                    from_txt=from_txt,
                                                    drop=self.__drop_table_before_creating)

        return query_dict

    def __form_scripts(self):
        self.__pay_vars = {var: [] for var in self.__pay_names}
        self.__pay_loan_vars = {var: [] for var in self.__pay_loan_names}
        self.__loan_vars = {var: [] for var in self.__loan_names}
        self.__var_vars = {var: [] for var in self.__var_names}
        self.__over_vars = []
        self.__ratio_vars = {var: [] for var in self.__ratio_names}

        query = {}
        query.update(self.__form_loans_and_payments_script())
        query.update(self.__form_vars_script())
        query.update(self.__form_overview_vars_script())
        query.update(self.__form_ratio_vars_script())

        for p in self.__processes:
            self.__query[p].update({q: re.sub(self.__suf, self.__processes[p], query[q]) for q in query})

    def create_scripts(self,
                       output_dir='FE_scripts',
                       split_scripts=False,
                       drop_table_before_creating=True,
                       use_compression=True):
        if not self.__table_for_each_variable and split_scripts:
            split_scripts = False
        self.__drop_table_before_creating = drop_table_before_creating
        self.__use_compression = use_compression
        self.__form_scripts()
        if output_dir not in os.listdir():
            os.mkdir(output_dir)
        for p in self.__processes:
            if p not in os.listdir(output_dir):
                os.mkdir(output_dir + '/' + p)
            if split_scripts is None:
                txt = [self.__query[p]['loans_and_payments']]
                if self.__table_for_each_variable:
                    for name in self.__pay_loan_names:
                        txt += [self.__query[p]['payments_agg|' + name]]
                    for name in self.__loan_names:
                        txt += [self.__query[p]['loans_agg|' + name]]
                else:
                    txt += [self.__query[p]['payments_agg']]
                    txt += [self.__query[p]['loans_agg']]
                txt += [self.__query[p]['overview']]
                if self.__table_for_each_variable:
                    for name in self.__ratio_names:
                        txt += [self.__query[p]['ratio|' + name]]
                else:
                    txt += [self.__query[p]['ratio']]
                open(output_dir + '/' + p + '/01_script.sql', 'w').write(('\n' * 8).join(txt))
            elif not split_scripts:
                txt = self.__query[p]['loans_and_payments']
                open(output_dir + '/' + p + '/01_create_loans_and_payments_script.sql', 'w').write(txt)

                txt = []
                if self.__table_for_each_variable:
                    for name in self.__pay_loan_names:
                        txt += [self.__query[p]['payments_agg|' + name]]
                    for name in self.__loan_names:
                        txt += [self.__query[p]['loans_agg|' + name]]
                else:
                    txt += [self.__query[p]['payments_agg']]
                    txt += [self.__query[p]['loans_agg']]
                open(output_dir + '/' + p + '/02_create_vars_script.sql', 'w').write(('\n' * 8).join(txt))

                txt = self.__query[p]['overview']
                open(output_dir + '/' + p + '/03_create_overview_vars_script.sql', 'w').write(txt)

                txt = []
                if self.__table_for_each_variable:
                    for name in self.__ratio_names:
                        txt += [self.__query[p]['ratio|' + name]]
                else:
                    txt += [self.__query[p]['ratio']]
                open(output_dir + '/' + p + '/04_create_ratio_vars_script.sql', 'w').write(('\n' * 8).join(txt))
            else:
                txt = self.__query[p]['loans_and_payments']
                open(output_dir + '/' + p + '/01_create_loans_and_payments_script.sql', 'w').write(txt)

                for name in self.__pay_loan_names:
                    txt = self.__query[p]['payments_agg|' + name]
                    open(output_dir + '/' + p + '/02_create_' + name + '_vars_script.sql', 'w').write(txt)
                for name in self.__loan_names:
                    txt = self.__query[p]['loans_agg|' + name]
                    open(output_dir + '/' + p + '/02_create_' + name + '_vars_script.sql', 'w').write(txt)

                txt = self.__query[p]['overview']
                open(output_dir + '/' + p + '/03_create_overview_vars_script.sql', 'w').write(txt)

                for name in self.__ratio_names:
                    txt = self.__query[p]['ratio|' + name]
                    open(output_dir + '/' + p + '/04_create_' + name + '_vars_script.sql', 'w').write(txt)

    def read_data(self,
                  ch,
                  base_schema_table,
                  base_table_id,
                  process,
                  join_how='inner',
                  reducing_memory_func=None):
        query = 'SELECT *\nFROM ' + base_schema_table
        data = pd.read_sql(query, ch)
        if reducing_memory_func is not None:
            data = reducing_memory_func(data)
        for name in self.__tables:
            if self.__table_for_each_variable:
                id = self.__loans_info['group_by'] if 'verview' not in name else self.__appl_info['id']
            else:
                id = self.__loans_info['group_by'] if 'agg' in name else self.__appl_info['id']
            query = 'SELECT n.*\nFROM ' + base_schema_table + ' AS b\n'
            query += 'INNER JOIN ' + self.__schema_prefix + name + self.__processes[process] + ' AS n\n'
            query += 'ON b.' + base_table_id + ' = n.' + id
            tmp_data = pd.read_sql(query, ch)
            if reducing_memory_func is not None:
                tmp_data = reducing_memory_func(tmp_data)
            tmp_data.columns = [base_table_id] + list(tmp_data.columns)[1:]
            data = data.merge(right=tmp_data,
                              how=join_how,
                              on=base_table_id)
            del tmp_data

        return data

    @staticmethod
    def feature_name(feature):
        return re.sub('_[A-Za-z]{3}[0-9]{2}', '', feature)

    @staticmethod
    def feature_fltr(feature):
        return [i[1:] for i in re.findall('_[A-Za-z]{3}[0-9]{2}', feature)]

    def feature_description(self, feature):
        if feature[-6:] != '_RATIO':
            fltr = self.feature_fltr(feature)
            name = self.feature_name(feature)
            if name in self.__over_names:
                description = self.__overview_vars_table[self.__overview_vars_table['name'] == name]['label'].values[0]
            elif name in self.__var_names:
                description = self.__vars_table[self.__vars_table['name'] == name]['label'].values[0]
                for flt in fltr:
                    if flt[-2:] != '00':
                        flt_descr = self.__flt_table[self.__flt_table['filter_id'] == flt][['category', 'label']].values
                        description += '. ' + flt_descr[0][0] + ': ' + flt_descr[0][1]
            else:
                description = 'superunknown'
        else:
            feat_split = feature[:-6].split('_TO_')
            if feat_split[0] + '_avg' == feat_split[1]:
                description = self.feature_description(feat_split[0]) + '. Нормировано'
            else:
                description = 'Отношение {' + self.feature_description(
                    feat_split[0]) + '} к {' + self.feature_description(
                    feat_split[1]) + '}'
        return description + '.'

    def features_description(self,
                             features,
                             n_threads=1):
        if isinstance(features, str):
            feat = [features]
        else:
            feat = features.copy()

        descr = pool_map(func=self.feature_description,
                         iterable=feat,
                         n_threads=n_threads)

        features_dict = {}
        for f, d in zip(feat, descr):
            features_dict[f] = d
        return features_dict

    def create_terms_of_reference(self,
                                  features):
        def is_ratio(feat_arr):
            return feat_arr.str[-6:] == '_RATIO'

        def is_name(feat_arr):
            return np.isin(feat_arr.apply(self.feature_name), self.__var_names + self.__ratio_names)

        def is_payname(feat, num=None):
            if num is None:
                if any(p in feat for p in self.__pay_names):
                    return feat
                else:
                    return ''
            elif num:
                if any(p in feat.split('_TO_')[0] for p in self.__pay_names):
                    return feat.split('_TO_')[0]
                else:
                    return ''
            else:
                if any(p in (feat[:-6].split('_TO_')[1] if feat[-6:] == '_RATIO' else '')
                       for p in self.__pay_names):
                    return feat[:-6].split('_TO_')[1] if feat[-6:] == '_RATIO' else ''
                else:
                    return ''

        def feature_pay_fltr(feature):
            return np.unique([i for i in self.feature_fltr(feature) if i[0] in ('p', 'P') and i[-2:] != '00']).tolist()

        def feature_loan_fltr(feature):
            return np.unique([i for i in self.feature_fltr(feature) if i[0] in ('c', 'C') and i[-2:] != '00']).tolist()

        def get_n_pay_flt(feat, n=1):
            if len(feature_pay_fltr(feat)) >= n:
                return feature_pay_fltr(feat)[n - 1]
            else:
                return ''

        def get_n_loan_flt(feat, n=1):
            if len(feature_loan_fltr(feat)) >= n:
                return feature_loan_fltr(feat)[n - 1]
            else:
                return ''

        def get_num(feat):
            return feat.split('_TO_')[0]

        def get_den(feat):
            return feat[:-6].split('_TO_')[1] if feat[-6:] == '_RATIO' else ''

        def common_flt(flt):
            flt_code = self.__flt_table[self.__flt_table['filter_id'] == flt]['code'].to_string(index=False)
            return flt + ' = IIF(' + flt_code + ', 1, 0)' if flt != '' else ''

        def payname_from_name(feat):
            return re.sub('_[A-Za-z]+$', '', re.sub('_C[A-Z]{2}[0-9]{2}|_c[a-z]{2}[0-9]{2}', '', feat))

        def code_from_name(feat, payvar='PX'):
            if self.feature_name(feat) not in self.__names + self.__pay_names:
                return feat
            if self.feature_name(feat) in self.__over_names:
                return self.__overview_vars_table[
                    self.__overview_vars_table['name'] == feat]['funs'].to_string(index=False)
            code = self.__vars_table[
                self.__vars_table['name'] == self.feature_name(feat)]['funs'].to_string(index=False)
            if self.feature_name(feat) in self.__pay_names:
                code = re.sub('flt', ' and '.join([i + ' = 1' for i in ['1'] + feature_pay_fltr(feat)]), code)
            else:
                code = re.sub('flt', ' and '.join([i + ' = 1' for i in ['1'] + feature_loan_fltr(feat)]), code)
                code = re.sub('|'.join(self.__pay_names), payvar, code)
            return code

        def drop_1eq1(code):
            return re.sub('\(1 = 1 and ', '(', re.sub(' and 1 = 1', '', code))

        pd.set_option('max_colwidth', 1024)

        tor = pd.DataFrame({'feature': features})
        tor['description'] = tor['feature'].apply(self.feature_description)

        max_pay_fltr = max([len(feature_pay_fltr(feature)) for feature in features])
        for i in range(1, max_pay_fltr + 1):
            tor['pay_flt' + str(i)] = np.where(is_name(tor['feature']),
                                               tor['feature'].apply(partial(get_n_pay_flt, n=i)).apply(common_flt),
                                               '')

        tor['PX'] = np.where(is_ratio(tor['feature']),
                             tor['feature'].apply(partial(is_payname, num=True)),
                             tor['feature'].apply(is_payname))
        tor['PX'] = tor['PX'].apply(payname_from_name).apply(code_from_name).apply(drop_1eq1)

        tor['PY'] = np.where(is_ratio(tor['feature']),
                             tor['feature'].apply(partial(is_payname, num=False)),
                             '')
        tor['PY'] = tor['PY'].apply(payname_from_name).apply(code_from_name).apply(drop_1eq1)

        max_loan_fltr = max([len(feature_loan_fltr(feature)) for feature in features])
        for i in range(1, max_loan_fltr + 1):
            tor['loan_flt' + str(i)] = np.where(is_name(tor['feature']),
                                                tor['feature'].apply(partial(get_n_loan_flt, n=i)).apply(common_flt),
                                                '')

        tor['X'] = np.where(is_ratio(tor['feature']),
                            tor['feature'].apply(get_num),
                            tor['feature'])
        tor['X'] = tor['X'].apply(partial(code_from_name, payvar='PX')).apply(drop_1eq1)

        tor['Y'] = np.where(is_ratio(tor['feature']),
                            tor['feature'].apply(get_den),
                            '')
        tor['Y'] = tor['Y'].apply(partial(code_from_name, payvar='PY')).apply(drop_1eq1)

        tor['var'] = np.where(is_ratio(tor['feature']), 'X / Y', 'X')
        return tor
