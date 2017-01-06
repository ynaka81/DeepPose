import os
import sys
import shutil
import paramiko
import scp

## Model file getter
#
# The getter function class of model file
class ModelFileGetter(object):
    ## constructor
    # @param local_model_dir The directory name which is used for local model file, the default value is 'tmp'
    # @param local_result_dir The local directory name of training result, the default value is 'result'
    # @param remote_result_dir The remote directory name of training result, the default value is '~/DeepPose/result'
    def __init__(self, local_model_dir="local_model", local_result_dir="result", remote_result_dir="~/DeepPose/result"):
        for k, v in locals().items():
            setattr(self, k, v)
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        try:
            os.makedirs(self.local_model_dir)
        except OSError:
            pass
    ## get model files
    # @param self The object pointer
    # @param locate Location of model definition file ('local':use local file, 'xxx.xxx.xxx.xxx':get remote file and use it)
    # @param model_class Class name of the model
    # @return Path to the directory containing model files
    def get(self, locate, model_class):
        # argument validation
        if not(len(locate) == len(model_class)):
            raise AttributeError("The length of attributes is not the same, locate({0}), model_class({1}).".format(len(locate), len(model_class)))
        # get model files
        N = len(locate)
        paths = []
        for i, (l, m) in enumerate(zip(locate, model_class)):
            sys.stderr.write("getting... {0} / {1}\r".format(i, N))
            sys.stderr.flush()
            path = os.path.join(self.local_model_dir, m)
            if l == "local":
                if not os.path.isdir(path):
                    raise IOError("The local model directory({0}) is not found.".format(path))
            elif l == "localhost":
                # remove old directory and get new one
                if os.path.exists(path):
                    shutil.rmtree(path)
                shutil.copytree(os.path.join(self.local_result_dir, m), path)
            else:
                user, host = l.split("@")
                self.ssh.connect(host, username=user)
                with scp.SCPClient(self.ssh.get_transport()) as scp_client:
                    # remove old directory and get new one
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    scp_client.get(os.path.join(self.remote_result_dir, m), path, recursive=True)
            paths.append(path)
        sys.stderr.write("getting... {0} / {1}\r".format(N, N))
        sys.stderr.write("\n")
        return paths
