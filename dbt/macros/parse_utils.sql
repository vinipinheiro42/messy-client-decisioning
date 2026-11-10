{% macro is_null_token(expr) %}
(
    {{ expr }} is null
    or trim(cast({{ expr }} as varchar)) = ''
    or lower(trim(cast({{ expr }} as varchar))) in ('n/a', '-', 'null', 'none')
)
{% endmacro %}


{% macro parse_brl(expr) %}
(
    case
        when {{ is_null_token(expr) }} then null
        else try_cast(
            case
                when regexp_matches(
                    replace(replace(replace(upper(trim(cast({{ expr }} as varchar))), 'R$', ''), ' ', ''), '+', ''),
                    '^-?[0-9]{1,3}(\.[0-9]{3})*,[0-9]{1,2}$'
                )
                or (
                    instr(cast({{ expr }} as varchar), ',') > 0
                    and (
                        instr(cast({{ expr }} as varchar), '.') = 0
                        or instr(cast({{ expr }} as varchar), ',')
                           > instr(cast({{ expr }} as varchar), '.')
                    )
                )
                then replace(replace(replace(replace(replace(cast({{ expr }} as varchar), 'R$', ''), 'r$', ''), ' ', ''), '.', ''), ',', '.')
                when instr(cast({{ expr }} as varchar), ',') > 0
                 and instr(cast({{ expr }} as varchar), '.') > 0
                 and instr(cast({{ expr }} as varchar), '.')
                     > instr(cast({{ expr }} as varchar), ',')
                then replace(replace(replace(replace(cast({{ expr }} as varchar), 'R$', ''), 'r$', ''), ' ', ''), ',', '')
                else replace(replace(replace(cast({{ expr }} as varchar), 'R$', ''), 'r$', ''), ' ', '')
            end as decimal(18, 2)
        )
    end
)
{% endmacro %}


{% macro parse_ts(expr) %}
(
    coalesce(
        try_strptime(trim(cast({{ expr }} as varchar)), '%Y-%m-%dT%H:%M:%SZ'),
        try_strptime(trim(cast({{ expr }} as varchar)), '%Y-%m-%d %H:%M:%S'),
        try_strptime(trim(cast({{ expr }} as varchar)), '%Y-%m-%d'),
        try_strptime(trim(cast({{ expr }} as varchar)), '%d/%m/%Y'),
        try_strptime(trim(cast({{ expr }} as varchar)), '%d-%m-%Y'),
        try_strptime(trim(cast({{ expr }} as varchar)), '%Y/%m/%d'),
        try_strptime(trim(cast({{ expr }} as varchar)), '%d.%m.%Y'),
        try_cast(trim(cast({{ expr }} as varchar)) as timestamp)
    )
)
{% endmacro %}


{% macro parse_date(expr) %}
cast({{ parse_ts(expr) }} as date)
{% endmacro %}


{% macro digits_only(expr) %}
regexp_replace(coalesce(cast({{ expr }} as varchar), ''), '[^0-9]', '', 'g')
{% endmacro %}


{% macro map_category(expr, column_name) %}
(
    select m.canonical_value
    from {{ ref('category_mappings') }} m
    where m.column_name = '{{ column_name }}'
      and lower(trim(m.raw_value)) = lower(trim(cast({{ expr }} as varchar)))
    limit 1
)
{% endmacro %}
