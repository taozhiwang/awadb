# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: awadb.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x61wadb.proto\x12\nawadb_grpc\"$\n\x06\x44\x42Name\x12\x11\n\x04name\x18\x01 \x01(\tH\x00\x88\x01\x01\x42\x07\n\x05_name\"\'\n\tTableName\x12\x11\n\x04name\x18\x01 \x01(\tH\x00\x88\x01\x01\x42\x07\n\x05_name\"r\n\x06\x44\x42Meta\x12\x14\n\x07\x64\x62_name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x11\n\x04\x64\x65sc\x18\x02 \x01(\tH\x01\x88\x01\x01\x12*\n\x0btables_meta\x18\x03 \x03(\x0b\x32\x15.awadb_grpc.TableMetaB\n\n\x08_db_nameB\x07\n\x05_desc\"\x1a\n\nTableNames\x12\x0c\n\x04name\x18\x01 \x03(\t\"o\n\tTableMeta\x12\x11\n\x04name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x11\n\x04\x64\x65sc\x18\x02 \x01(\tH\x01\x88\x01\x01\x12*\n\x0b\x66ields_meta\x18\x03 \x03(\x0b\x32\x15.awadb_grpc.FieldMetaB\x07\n\x05_nameB\x07\n\x05_desc\"\xe9\x01\n\nVectorMeta\x12-\n\tdata_type\x18\x01 \x01(\x0e\x32\x15.awadb_grpc.FieldTypeH\x00\x88\x01\x01\x12\x16\n\tdimension\x18\x02 \x01(\x05H\x01\x88\x01\x01\x12\x17\n\nstore_type\x18\x03 \x01(\tH\x02\x88\x01\x01\x12\x18\n\x0bstore_param\x18\x04 \x01(\tH\x03\x88\x01\x01\x12\x17\n\nhas_source\x18\x05 \x01(\x08H\x04\x88\x01\x01\x42\x0c\n\n_data_typeB\x0c\n\n_dimensionB\r\n\x0b_store_typeB\x0e\n\x0c_store_paramB\r\n\x0b_has_source\"\x80\x02\n\tFieldMeta\x12\x11\n\x04name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12(\n\x04type\x18\x02 \x01(\x0e\x32\x15.awadb_grpc.FieldTypeH\x01\x88\x01\x01\x12\x15\n\x08is_index\x18\x03 \x01(\x08H\x02\x88\x01\x01\x12\x15\n\x08is_store\x18\x04 \x01(\x08H\x03\x88\x01\x01\x12\x14\n\x07reindex\x18\x05 \x01(\x08H\x04\x88\x01\x01\x12-\n\x08vec_meta\x18\x06 \x01(\x0b\x32\x16.awadb_grpc.VectorMetaH\x05\x88\x01\x01\x42\x07\n\x05_nameB\x07\n\x05_typeB\x0b\n\t_is_indexB\x0b\n\t_is_storeB\n\n\x08_reindexB\x0b\n\t_vec_meta\"\xaa\x02\n\x0c\x44ocCondition\x12\x0f\n\x07\x64\x62_name\x18\x01 \x01(\t\x12\x12\n\ntable_name\x18\x02 \x01(\t\x12\x0b\n\x03ids\x18\x03 \x03(\t\x12\x41\n\rfilter_fields\x18\x04 \x03(\x0b\x32*.awadb_grpc.DocCondition.FilterFieldsEntry\x12\x1f\n\x12include_all_fields\x18\x05 \x01(\x08H\x00\x88\x01\x01\x12\x1a\n\x12not_include_fields\x18\x06 \x03(\t\x12\x12\n\x05limit\x18\x07 \x01(\x05H\x01\x88\x01\x01\x1a\x33\n\x11\x46ilterFieldsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42\x15\n\x13_include_all_fieldsB\x08\n\x06_limit\"\xab\x01\n\x05\x46ield\x12\x11\n\x04name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x12\n\x05value\x18\x02 \x01(\x0cH\x01\x88\x01\x01\x12(\n\x04type\x18\x03 \x01(\x0e\x32\x15.awadb_grpc.FieldTypeH\x02\x88\x01\x01\x12\x13\n\x06source\x18\x04 \x01(\tH\x03\x88\x01\x01\x12\x15\n\rmul_str_value\x18\x05 \x03(\tB\x07\n\x05_nameB\x08\n\x06_valueB\x07\n\x05_typeB\t\n\x07_source\"E\n\x08\x44ocument\x12\x0f\n\x02id\x18\x01 \x01(\x0cH\x00\x88\x01\x01\x12!\n\x06\x66ields\x18\x02 \x03(\x0b\x32\x11.awadb_grpc.FieldB\x05\n\x03_id\"y\n\tDocuments\x12\x14\n\x07\x64\x62_name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x17\n\ntable_name\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\"\n\x04\x64ocs\x18\x03 \x03(\x0b\x32\x14.awadb_grpc.DocumentB\n\n\x08_db_nameB\r\n\x0b_table_name\"v\n\nTermFilter\x12\x17\n\nfield_name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x12\n\x05value\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x15\n\x08is_union\x18\x03 \x01(\x05H\x02\x88\x01\x01\x42\r\n\x0b_field_nameB\x08\n\x06_valueB\x0b\n\t_is_union\"\xe5\x01\n\x0bRangeFilter\x12\x17\n\nfield_name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x18\n\x0blower_value\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x18\n\x0bupper_value\x18\x03 \x01(\tH\x02\x88\x01\x01\x12\x1a\n\rinclude_lower\x18\x04 \x01(\x08H\x03\x88\x01\x01\x12\x1a\n\rinclude_upper\x18\x05 \x01(\x08H\x04\x88\x01\x01\x42\r\n\x0b_field_nameB\x0e\n\x0c_lower_valueB\x0e\n\x0c_upper_valueB\x10\n\x0e_include_lowerB\x10\n\x0e_include_upper\"\x91\x02\n\x0bVectorQuery\x12\x17\n\nfield_name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x12\n\x05value\x18\x02 \x01(\x0cH\x01\x88\x01\x01\x12\x16\n\tmin_score\x18\x03 \x01(\x02H\x02\x88\x01\x01\x12\x16\n\tmax_score\x18\x04 \x01(\x02H\x03\x88\x01\x01\x12\x12\n\x05\x62oost\x18\x05 \x01(\x02H\x04\x88\x01\x01\x12\x15\n\x08is_boost\x18\x06 \x01(\x08H\x05\x88\x01\x01\x12\x1b\n\x0eretrieval_type\x18\x07 \x01(\tH\x06\x88\x01\x01\x42\r\n\x0b_field_nameB\x08\n\x06_valueB\x0c\n\n_min_scoreB\x0c\n\n_max_scoreB\x08\n\x06_boostB\x0b\n\t_is_boostB\x11\n\x0f_retrieval_type\"\x89\x04\n\rSearchRequest\x12\x14\n\x07\x64\x62_name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x17\n\ntable_name\x18\x02 \x01(\tH\x01\x88\x01\x01\x12,\n\x0bvec_queries\x18\x03 \x03(\x0b\x32\x17.awadb_grpc.VectorQuery\x12\x19\n\x11page_text_queries\x18\x04 \x03(\t\x12,\n\x0cterm_filters\x18\x05 \x03(\x0b\x32\x16.awadb_grpc.TermFilter\x12.\n\rrange_filters\x18\x06 \x03(\x0b\x32\x17.awadb_grpc.RangeFilter\x12\x11\n\x04topn\x18\x07 \x01(\x05H\x02\x88\x01\x01\x12\x1d\n\x10retrieval_params\x18\x08 \x01(\tH\x03\x88\x01\x01\x12\x1d\n\x10online_log_level\x18\t \x01(\tH\x04\x88\x01\x01\x12\x1f\n\x12\x62rute_force_search\x18\n \x01(\x08H\x05\x88\x01\x01\x12\x1f\n\x12is_pack_all_fields\x18\x0b \x01(\x08H\x06\x88\x01\x01\x12\x13\n\x0bpack_fields\x18\x0c \x03(\tB\n\n\x08_db_nameB\r\n\x0b_table_nameB\x07\n\x05_topnB\x13\n\x11_retrieval_paramsB\x13\n\x11_online_log_levelB\x15\n\x13_brute_force_searchB\x15\n\x13_is_pack_all_fields\">\n\nResultItem\x12\r\n\x05score\x18\x01 \x01(\x02\x12!\n\x06\x66ields\x18\x02 \x03(\x0b\x32\x11.awadb_grpc.Field\"t\n\x0cSearchResult\x12\x12\n\x05total\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12\x10\n\x03msg\x18\x02 \x01(\tH\x01\x88\x01\x01\x12,\n\x0cresult_items\x18\x03 \x03(\x0b\x32\x16.awadb_grpc.ResultItemB\x08\n\x06_totalB\x06\n\x04_msg\"\x85\x02\n\x0eSearchResponse\x12\x14\n\x07\x64\x62_name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x17\n\ntable_name\x18\x02 \x01(\tH\x01\x88\x01\x01\x12)\n\x07results\x18\x03 \x03(\x0b\x32\x18.awadb_grpc.SearchResult\x12\x1f\n\x12online_log_message\x18\x04 \x01(\tH\x02\x88\x01\x01\x12\x36\n\x0bresult_code\x18\x05 \x01(\x0e\x32\x1c.awadb_grpc.SearchResultCodeH\x03\x88\x01\x01\x42\n\n\x08_db_nameB\r\n\x0b_table_nameB\x15\n\x13_online_log_messageB\x0e\n\x0c_result_code\"M\n\x0eResponseStatus\x12&\n\x04\x63ode\x18\x01 \x01(\x0e\x32\x18.awadb_grpc.ResponseCode\x12\x13\n\x0boutput_info\x18\x02 \x01(\t*_\n\tFieldType\x12\x07\n\x03INT\x10\x00\x12\x08\n\x04LONG\x10\x01\x12\t\n\x05\x46LOAT\x10\x02\x12\n\n\x06\x44OUBLE\x10\x03\x12\n\n\x06STRING\x10\x04\x12\x10\n\x0cMULTI_STRING\x10\x05\x12\n\n\x06VECTOR\x10\x06*o\n\x10SearchResultCode\x12\x0b\n\x07SUCCESS\x10\x00\x12\x15\n\x11INDEX_NOT_TRAINED\x10\x01\x12\x10\n\x0cSEARCH_ERROR\x10\x02\x12\x10\n\x0c\x44\x42_NOT_FOUND\x10\x03\x12\x13\n\x0fTABLE_NOT_FOUND\x10\x04*j\n\x0cResponseCode\x12\x19\n\x15INPUT_PARAMETER_ERROR\x10\x00\x12\x07\n\x02OK\x10\xc8\x01\x12\r\n\x08TIME_OUT\x10\xc9\x01\x12\x13\n\x0eINTERNAL_ERROR\x10\xca\x01\x12\x12\n\rNETWORK_ERROR\x10\xcb\x01\x32\x82\x05\n\x0b\x41waDBServer\x12:\n\x06\x43reate\x12\x12.awadb_grpc.DBMeta\x1a\x1a.awadb_grpc.ResponseStatus\"\x00\x12:\n\x06\x44ropDB\x12\x12.awadb_grpc.DBName\x1a\x1a.awadb_grpc.ResponseStatus\"\x00\x12@\n\tDropTable\x12\x15.awadb_grpc.TableName\x1a\x1a.awadb_grpc.ResponseStatus\"\x00\x12:\n\nShowTables\x12\x12.awadb_grpc.DBName\x1a\x16.awadb_grpc.TableNames\"\x00\x12;\n\tDescTable\x12\x15.awadb_grpc.TableName\x1a\x15.awadb_grpc.TableMeta\"\x00\x12=\n\tAddFields\x12\x12.awadb_grpc.DBMeta\x1a\x1a.awadb_grpc.ResponseStatus\"\x00\x12\x42\n\x0b\x41\x64\x64OrUpdate\x12\x15.awadb_grpc.Documents\x1a\x1a.awadb_grpc.ResponseStatus\"\x00\x12\x38\n\x03Get\x12\x18.awadb_grpc.DocCondition\x1a\x15.awadb_grpc.Documents\"\x00\x12\x41\n\x06Search\x12\x19.awadb_grpc.SearchRequest\x1a\x1a.awadb_grpc.SearchResponse\"\x00\x12@\n\x06\x44\x65lete\x12\x18.awadb_grpc.DocCondition\x1a\x1a.awadb_grpc.ResponseStatus\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'awadb_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DOCCONDITION_FILTERFIELDSENTRY._options = None
  _DOCCONDITION_FILTERFIELDSENTRY._serialized_options = b'8\001'
  _globals['_FIELDTYPE']._serialized_start=3204
  _globals['_FIELDTYPE']._serialized_end=3299
  _globals['_SEARCHRESULTCODE']._serialized_start=3301
  _globals['_SEARCHRESULTCODE']._serialized_end=3412
  _globals['_RESPONSECODE']._serialized_start=3414
  _globals['_RESPONSECODE']._serialized_end=3520
  _globals['_DBNAME']._serialized_start=27
  _globals['_DBNAME']._serialized_end=63
  _globals['_TABLENAME']._serialized_start=65
  _globals['_TABLENAME']._serialized_end=104
  _globals['_DBMETA']._serialized_start=106
  _globals['_DBMETA']._serialized_end=220
  _globals['_TABLENAMES']._serialized_start=222
  _globals['_TABLENAMES']._serialized_end=248
  _globals['_TABLEMETA']._serialized_start=250
  _globals['_TABLEMETA']._serialized_end=361
  _globals['_VECTORMETA']._serialized_start=364
  _globals['_VECTORMETA']._serialized_end=597
  _globals['_FIELDMETA']._serialized_start=600
  _globals['_FIELDMETA']._serialized_end=856
  _globals['_DOCCONDITION']._serialized_start=859
  _globals['_DOCCONDITION']._serialized_end=1157
  _globals['_DOCCONDITION_FILTERFIELDSENTRY']._serialized_start=1073
  _globals['_DOCCONDITION_FILTERFIELDSENTRY']._serialized_end=1124
  _globals['_FIELD']._serialized_start=1160
  _globals['_FIELD']._serialized_end=1331
  _globals['_DOCUMENT']._serialized_start=1333
  _globals['_DOCUMENT']._serialized_end=1402
  _globals['_DOCUMENTS']._serialized_start=1404
  _globals['_DOCUMENTS']._serialized_end=1525
  _globals['_TERMFILTER']._serialized_start=1527
  _globals['_TERMFILTER']._serialized_end=1645
  _globals['_RANGEFILTER']._serialized_start=1648
  _globals['_RANGEFILTER']._serialized_end=1877
  _globals['_VECTORQUERY']._serialized_start=1880
  _globals['_VECTORQUERY']._serialized_end=2153
  _globals['_SEARCHREQUEST']._serialized_start=2156
  _globals['_SEARCHREQUEST']._serialized_end=2677
  _globals['_RESULTITEM']._serialized_start=2679
  _globals['_RESULTITEM']._serialized_end=2741
  _globals['_SEARCHRESULT']._serialized_start=2743
  _globals['_SEARCHRESULT']._serialized_end=2859
  _globals['_SEARCHRESPONSE']._serialized_start=2862
  _globals['_SEARCHRESPONSE']._serialized_end=3123
  _globals['_RESPONSESTATUS']._serialized_start=3125
  _globals['_RESPONSESTATUS']._serialized_end=3202
  _globals['_AWADBSERVER']._serialized_start=3523
  _globals['_AWADBSERVER']._serialized_end=4165
# @@protoc_insertion_point(module_scope)
