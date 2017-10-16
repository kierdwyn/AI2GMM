projectPath = '../';
utilPath = '../matlab_utils/';

addpath([utilPath 'GetFullPath']);
try
    GetFullPath('.');
catch ME
    delete([utilPath 'GetFullPath/GetFullPath.mexw64']);
    InstallMex GetFullPath.c
end

utilPath = GetFullPath(utilPath);
addpath([utilPath 'utils']);
addpath([utilPath 'i2m_fileio']);
addpath([utilPath 'liblinear-2.1/windows']);
addpath(GetFullPath([projectPath 'matlab']));
