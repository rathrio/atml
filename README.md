Using the cluster
=================

First, make sure that you are in the Unibe network. Use a [VPN][0] if necessary.

1. ssh into the cluster (provide the password from the screenshot in the
   whatsapp group)

   ```
   $ ssh studi9@cluster.inf.unibe.ch
   ```
2. ssh into node05

   ```
   $ ssh node05
   ```
3. Load the correct python environment

   ```
   $ module load anaconda/3
   $ source activate /var/tmp/group5/
   ```

Done!

[0]: http://www.unibe.ch/university/campus_and_infrastructure/rund_um_computer/internetzugang/access_to_internal_resources_via_vpn/index_eng.html
