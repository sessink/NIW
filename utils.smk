rule utils:
    input:
        expand("viz/{graph}.{fmt}", graph=["rulegraph", "dag"], fmt=["pdf", "png"])

rule graph:
    output:
        "viz/{graph,(dag|rulegraph)}.dot"
    shell:
        "snakemake --{wildcards.graph} > {output}"

rule render_dot:
    input:
        "{prefix}.dot"
    output:
        "{prefix}.{fmt,(png|pdf)}"
    shell:
        "dot -T{wildcards.fmt} < {input} > {output}"
