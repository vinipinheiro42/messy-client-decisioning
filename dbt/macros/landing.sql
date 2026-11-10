{% macro landing_parquet(entity) %}
read_parquet('{{ var("landing_dir", "../data/bronze") }}/{{ entity }}.parquet')
{% endmacro %}
