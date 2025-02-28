FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Remove #s for the two lines below after setting your time zone, so that you can avoid an interruption in the process of building the environment.
#ENV TZ={your time zone here}
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    screen \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    texlive-science \
    curl

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN pip install absl-py==2.1.0 \
                accelerate==1.1.1 \
                aiohappyeyeballs==2.4.3 \
                aiohttp==3.11.7 \
                aiosignal==1.3.1 \
                annotated-types==0.7.0 \
                anthropic==0.39.0 \
                anyio==4.6.2.post1 \
                arxiv==2.1.3 \
                astunparse==1.6.3 \
                async-timeout==5.0.1 \
                attrs==24.2.0 \
                blis==1.0.1 \
                catalogue==2.0.10 \
                certifi==2024.8.30 \
                charset-normalizer==3.4.0 \
                click==8.1.7 \
                cloudpathlib==0.20.0 \
                confection==0.1.5 \
                contourpy==1.3.0 \
                cycler==0.12.1 \
                cymem==2.0.10 \
                datasets==3.1.0 \
                diffusers==0.31.0 \
                dill==0.3.8 \
                distro==1.9.0 \
                exceptiongroup==1.2.2 \
                feedparser==6.0.11 \
                filelock==3.16.1 \
                flatbuffers==24.3.25 \
                fonttools==4.55.0 \
                frozenlist==1.5.0 \
                fsspec==2024.9.0 \
                gast==0.6.0 \
                google-pasta==0.2.0 \
                grpcio==1.68.0 \
                h11==0.14.0 \
                h5py==3.12.1 \
                httpcore==1.0.7 \
                httpx==0.27.2 \
                huggingface-hub==0.26.2 \
                idna==3.10 \
                imageio==2.36.0 \
                importlib_metadata==8.5.0 \
                importlib_resources==6.4.5 \
                Jinja2==3.1.4 \
                jiter==0.7.1 \
                joblib==1.4.2 \
                kiwisolver==1.4.7 \
                langcodes==3.5.0 \
                language_data==1.3.0 \
                lazy_loader==0.4 \
                libclang==18.1.1 \
                marisa-trie==1.2.1 \
                Markdown==3.7 \
                markdown-it-py==3.0.0 \
                MarkupSafe==3.0.2 \
                matplotlib==3.9.2 \
                mdurl==0.1.2 \
                ml-dtypes==0.4.1 \
                mpmath==1.3.0 \
                multidict==6.1.0 \
                multiprocess==0.70.16 \
                murmurhash==1.0.11 \
                namex==0.0.8 \
                nest-asyncio==1.6.0 \
                networkx==3.2.1 \
                nltk==3.9.1 \
                numpy==2.0.2 \
                openai==1.55.1 \
                opt_einsum==3.4.0 \
                optree==0.13.1 \
                packaging==24.2 \
                pandas==2.2.3 \
                patsy==1.0.1 \
                pillow==11.0.0 \
                plotly==5.24.1 \
                preshed==3.0.9 \
                propcache==0.2.0 \
                protobuf==5.28.3 \
                psutil==6.1.0 \
                pyarrow==18.1.0 \
                pydantic==2.10.2 \
                pydantic_core==2.27.1 \
                Pygments==2.18.0 \
                pyparsing==3.2.0 \
                pypdf==5.1.0 \
                python-dateutil==2.9.0.post0 \
                pytz==2024.2 \
                PyYAML==6.0.2 \
                regex==2024.11.6 \
                requests==2.32.3 \
                rich==13.9.4 \
                sacremoses==0.1.1 \
                safetensors==0.4.5 \
                scikit-image==0.24.0 \
                scikit-learn==1.5.2 \
                scipy==1.13.1 \
                seaborn==0.13.2 \
                semanticscholar==0.8.4 \
                sgmllib3k==1.0.0 \
                shellingham==1.5.4 \
                six==1.16.0 \
                smart-open==7.0.5 \
                sniffio==1.3.1 \
                spacy==3.8.2 \
                spacy-legacy==3.0.12 \
                spacy-loggers==1.0.5 \
                srsly==2.4.8 \
                statsmodels==0.14.4 \
                sympy==1.13.1 \
                tenacity==9.0.0 \
                tensorboard==2.18.0 \
                termcolor==2.5.0 \
                thinc==8.3.2 \
                threadpoolctl==3.5.0 \
                tifffile==2024.8.30 \
                tiktoken==0.8.0 \
                tokenizers==0.20.4 \
                tqdm==4.67.1 \
                transformers==4.46.3 \
                typer==0.13.1 \
                typing_extensions==4.12.2 \
                tzdata==2024.2 \
                urllib3==2.2.3 \
                wasabi==1.1.3 \
                weasel==0.4.1 \
                Werkzeug==3.1.3 \
                wrapt==1.17.0 \
                xxhash==3.5.0 \
                yarl==1.18.0 \
                zipp==3.21.0


RUN mkdir /work && chmod 777 /work
WORKDIR /work