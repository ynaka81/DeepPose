import os
import shutil
import tempfile
import paramiko
import scp
import argparse
import json
import matplotlib.pylab as plt

## Visualizing log
#
# The log visualization tool
class VisualizeLog(object):
    ## constructor
    # @param args The command line arguments
    def __init__(self, args):
        self.args = args
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
    ## main method of generating the HumanEva dataset
    # @param self The object pointer
    def main(self):
        # make temporary directory for log
        temp = tempfile.mkdtemp()
        # get log
        if self.args.locate == "localhost":
            shutil.copy(os.path.join(self.args.local, self.args.model, "log"), temp)
        else:
            user, host = l.split("@")
            self.ssh.connect(host, username=user)
            with scp.SCPClient(self.ssh.get_transport()) as scp_client:
                scp_client.get(os.path.join(self.args.remote, self.args.model, "log"), temp)
        # read log
        train_epoch = []
        train_loss = []
        val_epoch = []
        val_loss = []
        for data in json.load(open(os.path.join(temp, "log"))):
            train_epoch.append(data["epoch"])
            train_loss.append(data["main/loss"])
            if "validation/main/loss" in data:
                val_epoch.append(data["epoch"])
                val_loss.append(data["validation/main/loss"])
        # remove temp directory
        shutil.rmtree(temp)
        # draw graph
        plt.plot(train_epoch, train_loss, label="train")
        plt.plot(val_epoch, val_loss, label="validation")
        plt.yscale("log")
        plt.legend()
        plt.title("loss function value per epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

if __name__ == "__main__":
    # arg definition
    parser = argparse.ArgumentParser(description="Visualizing log")
    parser.add_argument("--locate", "-l", type=str, default="localhost", help="Location of model definition file ('localhost':get local file and use it, 'username@xxx.xxx.xxx.xxx':get remote file and use it)")
    parser.add_argument("--model", "-m", type=str, default="AlexNet", help="Model name to visualize log")
    parser.add_argument("--local", type=str, default="result", help="The local directory name of training result")
    parser.add_argument("--remote", type=str, default="~/DeepPose/result", help="The remote directory name of training result")
    VisualizeLog(parser.parse_args()).main()
