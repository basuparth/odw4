from colorama import Fore, Back, Style
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter
import numpy as np
from pycbc.vetoes import power_chisq
from pycbc.events.ranking import newsnr
import matplotlib.pyplot as plt
import pandas as pd

masses = []
for x in range(32, 33):
    print(
        color.BOLD + "Individual masses of the BBHs -- ", x, "solar masses" + color.END
    )
    print(Style.RESET_ALL)

    hp1, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=x,
        mass2=x,
        delta_f=data_H1.delta_f,
        f_lower=20.0,
    )

    hp1.resize(len(psd_H1))

    snr1 = matched_filter(hp1, data_H1, psd=psd_H1, low_frequency_cutoff=20)

    snr1 = snr1.crop(5, 4)
    peak1 = abs(snr1).numpy().argmax()
    snrp1 = snr1[peak1]
    time1 = snr1.sample_times[peak1]

    # print("We found a possible signal candidate at {}s with SNR {} in H1".format(time1,
    #                                                    abs(snrp1)))
    hp2, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=x,
        mass2=x,
        delta_f=data_L1.delta_f,
        f_lower=20.0,
    )

    # We will resize the vector to match our data
    hp2.resize(len(psd_L1))

    snr2 = matched_filter(hp2, data_L1, psd=psd_L1, low_frequency_cutoff=20)

    snr2 = snr2.crop(5, 4)
    peak2 = abs(snr2).numpy().argmax()
    snrp2 = snr2[peak2]
    time2 = snr2.sample_times[peak2]

    # print("We found a possible signal candidate at {}s with SNR {} in L1".format(time2,
    #                                                    abs(snrp2)))
    # if time1==time2:
    # print("Peak Time is coincident in both detectors")

    plt.figure(figsize=[10, 4])
    plt.plot(snr1.sample_times, abs(snr1), "-b")
    plt.title("All Possible Signal Candidates in H1")
    plt.ylabel("Signal-to-noise")
    plt.xlabel("Time (s)")
    plt.show()

    df_1 = pd.DataFrame({"snr1_x": snr1.sample_times, "snr1_y": abs(snr1).numpy()})
    # df.plot('x', 'y', kind='scatter')
    BBH_dg = 750  # Assuming that for any apparent signal detected there will be no other merger
    # peak within .36secs on either side of the said tentative detection as it is implausible

    maxv_snr1 = df_1["snr1_y"].max()
    max_i1 = df_1["snr1_y"].idxmax()
    snr1_xmax = df_1["snr1_x"][max_i1]
    snr1_ymax = df_1["snr1_y"][max_i1]
    # print(maxv_snr1)
    # print(max_i1)
    # print(snr1_xmax)
    # print(snr1_ymax)

    if maxv_snr1 >= 8:
        print(
            color.BLUE + "Listing all the possible signal candidates(for BH mass=",
            x,
            "solar mass) at Hanford in order of decreasing SNR" + color.END,
        )
        print(Style.RESET_ALL)
        while maxv_snr1 >= 8:
            print(
                "We found a possible signal candidate in H1 at",
                snr1_xmax,
                "s with SNR",
                snr1_ymax,
            )

            df_1 = df_1.drop(labels=range(max_i1 - BBH_dg, max_i1 + BBH_dg), axis=0)
            maxv_snr1 = df_1["snr1_y"].max()
            max_i1 = df_1["snr1_y"].idxmax()
            snr1_xmax = df_1["snr1_x"][max_i1]
            snr1_ymax = df_1["snr1_y"][max_i1]
            # print(maxv_snr1)
            # print(max_i1)
            # print(snr1_xmax)
            # print(snr1_ymax)
    else:
        print("There were no signal candidates at Hanford")
        print(Style.RESET_ALL)

    plt.figure(figsize=[10, 4])
    plt.plot(snr2.sample_times, abs(snr2), "-g")
    plt.title("All Possible Signal Candidates in L1")
    plt.ylabel("Signal-to-noise")
    plt.xlabel("Time (s)")
    plt.show()

    df_2 = pd.DataFrame({"snr2_x": snr2.sample_times, "snr2_y": abs(snr2).numpy()})
    # df.plot('x', 'y', kind='scatter')
    BBH_dg = 750  # Assuming that for any apparent signal detected there will be no other merger
    # peak within .36secs on either side of the said tentative detection as it is implausible

    maxv_snr2 = df_2["snr2_y"].max()
    max_i2 = df_2["snr2_y"].idxmax()
    snr2_xmax = df_2["snr2_x"][max_i2]
    snr2_ymax = df_2["snr2_y"][max_i2]
    # print(maxv_snr1)
    # print(max_i)
    # print(snr1_xmax)
    # print(snr1_ymax)

    if maxv_snr2 >= 8:
        print(
            color.GREEN + "Listing all the possible signal candidates(for BH mass=",
            x,
            "solar mass) at Livingston in order of decreasing SNR" + color.END,
        )
        print(Style.RESET_ALL)
        while maxv_snr2 >= 8:
            print(
                "We found a possible signal candidate in L1 at",
                snr2_xmax,
                "s with SNR",
                snr2_ymax,
            )

            df_2 = df_2.drop(labels=range(max_i2 - BBH_dg, max_i2 + BBH_dg), axis=0)
            maxv_snr2 = df_2["snr2_y"].max()
            max_i2 = df_2["snr2_y"].idxmax()
            snr2_xmax = df_2["snr2_x"][max_i2]
            snr2_ymax = df_2["snr2_y"][max_i2]
            # print(maxv_snr1)
            # print(max_i)
            # print(snr1_xmax)
            # print(snr1_ymax)
    else:
        print("There were no signal candidates at Livingston")
        print(Style.RESET_ALL)

    nbins = 26
    dof = nbins * 2 - 2
    chisq_H1 = power_chisq(hp1, data_H1, nbins, psd_H1, low_frequency_cutoff=20.0)
    chisq_H1 = chisq_H1.crop(5, 4)
    chisq_H1 /= dof

    chisq_L1 = power_chisq(hp2, data_L1, nbins, psd_L1, low_frequency_cutoff=20.0)
    chisq_L1 = chisq_L1.crop(5, 4)
    chisq_L1 /= dof

    nsnr1 = newsnr(abs(snr1), chisq_H1)
    nsnr2 = newsnr(abs(snr2), chisq_L1)

    peak_H1 = nsnr1.argmax()
    snrp_H1 = nsnr1[peak_H1]
    time_H1 = snr1.sample_times[peak_H1]

    if snrp_H1 > 8:
        print(
            "For the Hanford data we found a confirmed signal at {}s with SNR {} after ruling out the other candidates as glitches".format(
                time_H1, abs(snrp_H1)
            )
        )
        print(Style.RESET_ALL)
        plt.figure(figsize=[10, 4])
        plt.plot(snr1.sample_times, nsnr1, "-b")
        plt.title("Remaining Signal Candidate in H1 after reweighing data")
        plt.xlabel("Time (s)")
        plt.ylabel("Re-weighted Signal-to-noise")
        plt.show()

    peak_L1 = nsnr2.argmax()
    snrp_L1 = nsnr2[peak_L1]
    time_L1 = snr2.sample_times[peak_L1]

    if snrp_L1 > 8:
        print(
            "For the Livingston data we found a confirmed signal at {}s with SNR {} after ruling out the other candidates as glitches".format(
                time_L1, abs(snrp_L1)
            )
        )
        print(Style.RESET_ALL)
        plt.figure(figsize=[10, 4])
        plt.plot(snr2.sample_times, nsnr2, "-g")
        plt.title("Remaining Signal Candidate in L1 after reweighing data")
        plt.xlabel("Time (s)")
        plt.ylabel("Re-weighted Signal-to-noise")
        plt.show()

    ############################################################
    ############################################################
    ndf_1 = pd.DataFrame({"x1": snr1.sample_times, "y1": nsnr1})
    ndf_2 = pd.DataFrame({"x2": snr2.sample_times, "y2": nsnr2})

    BBH_dg = 750  # Assuming that for any apparent signal detected there will be no other merger
    # peak within .36secs on either side of the said tentative detection as it is implausible

    max_value1 = ndf_1["y1"].max()
    max_index1 = ndf_1["y1"].idxmax()
    x_max1 = ndf_1["x1"][max_index1]
    y_max1 = ndf_1["y1"][max_index1]

    max_value2 = ndf_2["y2"].max()
    max_index2 = ndf_2["y2"].idxmax()
    x_max2 = ndf_2["x2"][max_index2]
    y_max2 = ndf_2["y2"][max_index2]

    if max_value1 >= 8:
        # print(color.BOLD + "Listing all the remaining signal candidate(s) at Hanford" + color.END)
        print(Style.RESET_ALL)
        x_max1l = []
        y_max1l = []
        while max_value1 >= 8:
            # print("We found a confirmed signal candidate in H1 at", x_max1,"s with SNR", y_max1)

            x_max1l.append(x_max1)
            y_max1l.append(y_max1)
            ndf_1 = ndf_1.drop(
                labels=range(max_index1 - BBH_dg, max_index1 + BBH_dg), axis=0
            )
            max_value1 = ndf_1["y1"].max()
            max_index1 = ndf_1["y1"].idxmax()
            x_max1 = ndf_1["x1"][max_index1]
            y_max1 = ndf_1["y1"][max_index1]

        print(Style.RESET_ALL)

        x_max1ar = np.asarray(x_max1l)
        y_max1ar = np.asarray(y_max1l)
        dfn_1 = (
            pd.DataFrame({"X1": x_max1ar, "Y1": y_max1ar})
            .sort_values("X1")
            .reset_index()
        )

    else:
        print("There were no signal candidates at Hanford")
        print(Style.RESET_ALL)

    if max_value2 >= 8:
        # print(color.BOLD + "Listing all the remaining signal candidate(s) at Livingston" + color.END)
        print(Style.RESET_ALL)
        x_max2l = []
        y_max2l = []
        while max_value2 >= 8:
            # print("We found a confirmed signal candidate in L1 at", x_max2,"s with SNR", y_max2)

            x_max2l.append(x_max2)
            y_max2l.append(y_max2)
            ndf_2 = ndf_2.drop(
                labels=range(max_index2 - BBH_dg, max_index2 + BBH_dg), axis=0
            )
            max_value2 = ndf_2["y2"].max()
            max_index2 = ndf_2["y2"].idxmax()
            x_max2 = ndf_2["x2"][max_index2]
            y_max2 = ndf_2["y2"][max_index2]

        print(Style.RESET_ALL)

        x_max2ar = np.asarray(x_max2l)
        y_max2ar = np.asarray(y_max2l)
        dfn_2 = (
            pd.DataFrame({"X2": x_max2ar, "Y2": y_max2ar})
            .sort_values("X2")
            .reset_index()
        )

    else:
        print("There were no signal candidates at Livingston")
        print(Style.RESET_ALL)

    # dfn_c = pd.concat([dfn_1, dfn_2], axis=1)
    # dfn_cn = dfn_c.loc[dfn_1['X1'].astype(int).isin(dfn_2['X2'].astype(int))]

    if len(dfn_1) == 0 and len(dfn_2) == 0:
        print("There are no remaining signal candidates in both H1 and L1")

    elif len(dfn_1) > 0 and len(dfn_2) == 0:
        print(
            "There are no surviving signal candidates in L1 whereas for H1 we have",
            len(dfn_1),
            "candidates:",
        )
        for i in range(0, dfn_1.index[-1] + 1):
            print(
                "The signal candidate in H1 at",
                dfn_1["X1"][i],
                "has a SNR of",
                dfn_1["Y1"][i],
            )

    elif len(dfn_2) > 0 and len(dfn_1) == 0:
        print(
            "There are no surviving signal candidates in H1 whereas for L1 we have",
            len(dfn_2),
            "candidates:",
        )
        for j in range(0, dfn_2.index[-1] + 1):
            print(
                "The signal candidate in L1 at",
                dfn_2["X2"][j],
                "has a SNR of",
                dfn_2["Y2"][j],
            )

    elif (len(dfn_2) > 0 and len(dfn_1) > 0) and (len(dfn_2) == len(dfn_1)):
        print(Style.RESET_ALL)
        print(color.BLUEB + "The surviving signal candidates in H1 are:" + color.END)
        print(Style.RESET_ALL)
        for i in range(0, dfn_1.index[-1] + 1):
            print(
                "There is a signal candidate in H1 at",
                dfn_1["X1"][i],
                "with SNR",
                dfn_1["Y1"][i],
            )

        print(Style.RESET_ALL)
        print(color.GREENB + "The surviving signal candidates in L1 are:" + color.END)
        print(Style.RESET_ALL)
        for j in range(0, dfn_2.index[-1] + 1):
            print(
                "There is a signal candidate in L1 at",
                dfn_2["X2"][j],
                "with SNR",
                dfn_2["Y2"][j],
            )

        print(Style.RESET_ALL)
        print(
            color.BOLD
            + "Checking if the surviving signal candidates fit the criteria for a BBH merger for our chosen template:"
            + color.END
        )
        # print(Style.RESET_ALL)

        for i in range(0, dfn_1.index[-1] + 1):
            for j in range(0, dfn_2.index[-1] + 1):

                if dfn_1["X1"][i] == dfn_2["X2"][j]:
                    print(Style.RESET_ALL)
                    print(
                        "As the timestamps are exactly same we conclude an injected signal mimicking a BBH merger at",
                        dfn_1["X1"][i],
                        "\nwith the H1 SNR",
                        dfn_1["Y1"][i],
                        "and the L1 SNR",
                        dfn_2["Y2"][j],
                    )
                    print(Style.RESET_ALL)

                elif (
                    dfn_2["X2"][j] - 0.008961194
                    <= dfn_1["X1"][i]
                    <= dfn_2["X2"][j] + 0.008961194
                ):

                    print(Style.RESET_ALL)
                    print(
                        "We found a BBH merger candidate \nin H1 at",
                        dfn_1["X1"][i],
                        "with SNR",
                        dfn_1["Y1"][i],
                        "and \nin L1 at",
                        dfn_2["X2"][j],
                        "with SNR",
                        dfn_2["Y2"][j],
                    )
                    print(Style.RESET_ALL)

                    if abs(dfn_2["X2"][j] - dfn_1["X1"][i]) < 0.008961194:
                        # print(Style.RESET_ALL)
                        print(
                            "The time lag",
                            abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                            "s between the signals at H1 & L1 is too short compared to GW travel time and hence \nan artefact of other noise sources or some artificial source",
                        )
                        print(Style.RESET_ALL)

                    else:
                        print(color.BLUEB + "yipee" + color.END)

                elif dfn_2["X2"][j] - 3 <= dfn_1["X1"][i] <= dfn_2["X2"][j] + 3:

                    if abs(dfn_2["X2"][j] - dfn_1["X1"][i]) > 0.011328301:
                        print(Style.RESET_ALL)
                        print(
                            "While the time lag",
                            abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                            "s is short it is too large compared to GW travel time and hence an artefact of other noise sources",
                        )
                        print(Style.RESET_ALL)
                    elif (abs(dfn_2["X2"][j] - dfn_1["X1"][i]) >= 0.008961194) and (
                        abs(dfn_2["X2"][j] - dfn_1["X1"][i]) < 0.011328301
                    ):
                        print(Style.RESET_ALL)
                        print(
                            "We found a signal candidate \nin H1 at",
                            dfn_1["X1"][i],
                            "with SNR",
                            dfn_1["Y1"][i],
                            "and \nin L1 at",
                            dfn_2["X2"][j],
                            "with SNR",
                            dfn_2["Y2"][j],
                        )
                        print(
                            "The above signal is an extremely likely candidate for an authentic BBH merger given the choice of template"
                        )
                        print(
                            "The time lag",
                            abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                            "s between the signals at H1 & L1 is comparable to GW travel time",
                        )
                        print(Style.RESET_ALL)
                    else:
                        print(color.BLUEB + "yipee" + color.END)

                elif abs(dfn_2["X2"][j] - dfn_1["X1"][i]) > 3:

                    print(
                        "The signals are too far apart with a lag of",
                        abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                        " s",
                    )

    elif (len(dfn_2) > 0 and len(dfn_1) > 0) and (len(dfn_2) != len(dfn_1)):
        print(Style.RESET_ALL)
        print(color.BLUEB + "The surviving signal candidates in H1 are:" + color.END)
        print(Style.RESET_ALL)
        for i in range(0, dfn_1.index[-1] + 1):
            print(
                "There is a signal candidate in H1 at",
                dfn_1["X1"][i],
                "with SNR",
                dfn_1["Y1"][i],
            )

        print(Style.RESET_ALL)
        print(color.GREENB + "The surviving signal candidates in L1 are:" + color.END)
        print(Style.RESET_ALL)
        for j in range(0, dfn_2.index[-1] + 1):
            print(
                "There is a signal candidate in L1 at",
                dfn_2["X2"][j],
                "with SNR",
                dfn_2["Y2"][j],
            )

        print(Style.RESET_ALL)
        print(
            color.BOLD
            + "Checking if the surviving signal candidates fit the criteria for a BBH merger for our chosen template:"
            + color.END
        )
        # print(Style.RESET_ALL)

        for i in range(0, dfn_1.index[-1] + 1):
            for j in range(0, dfn_2.index[-1] + 1):

                if dfn_1["X1"][i] == dfn_2["X2"][j]:
                    print(Style.RESET_ALL)
                    print(
                        "As the timestamps are exactly same we conclude an injected signal mimicking a BBH merger at",
                        dfn_1["X1"][i],
                        "\nwith the H1 SNR",
                        dfn_1["Y1"][i],
                        "and the L1 SNR",
                        dfn_2["Y2"][j],
                    )
                    print(Style.RESET_ALL)

                elif (
                    dfn_2["X2"][j] - 0.008961194
                    <= dfn_1["X1"][i]
                    <= dfn_2["X2"][j] + 0.008961194
                ):

                    print(Style.RESET_ALL)
                    print(
                        "We found a BBH merger candidate \nin H1 at",
                        dfn_1["X1"][i],
                        "with SNR",
                        dfn_1["Y1"][i],
                        "and \nin L1 at",
                        dfn_2["X2"][j],
                        "with SNR",
                        dfn_2["Y2"][j],
                    )
                    print(Style.RESET_ALL)

                    if abs(dfn_2["X2"][j] - dfn_1["X1"][i]) < 0.008961194:
                        # print(Style.RESET_ALL)
                        print(
                            "The time lag",
                            abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                            "s between the signals at H1 & L1 is too short compared to GW travel time and hence \nan artefact of other noise sources or some artificial source",
                        )
                        print(Style.RESET_ALL)

                    else:
                        print(color.BLUEB + "yipee" + color.END)

                elif dfn_2["X2"][j] - 3 <= dfn_1["X1"][i] <= dfn_2["X2"][j] + 3:

                    if abs(dfn_2["X2"][j] - dfn_1["X1"][i]) > 0.011328301:
                        print(Style.RESET_ALL)
                        print(
                            "While the time lag",
                            abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                            "s is short it is too large compared to GW travel time and hence an artefact of other noise sources",
                        )
                        print(Style.RESET_ALL)
                    elif (abs(dfn_2["X2"][j] - dfn_1["X1"][i]) >= 0.008961194) and (
                        abs(dfn_2["X2"][j] - dfn_1["X1"][i]) < 0.011328301
                    ):
                        print(Style.RESET_ALL)
                        print(
                            "We found a signal candidate \nin H1 at",
                            dfn_1["X1"][i],
                            "with SNR",
                            dfn_1["Y1"][i],
                            "and \nin L1 at",
                            dfn_2["X2"][j],
                            "with SNR",
                            dfn_2["Y2"][j],
                        )
                        print(
                            "The above signal is an extremely likely candidate for an authentic BBH merger given the choice of template"
                        )
                        print(
                            "The time lag",
                            abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                            "s between the signals at H1 & L1 is comparable to GW travel time",
                        )
                        print(Style.RESET_ALL)
                    else:
                        print(color.BLUEB + "yipee" + color.END)

                elif abs(dfn_2["X2"][j] - dfn_1["X1"][i]) > 3:

                    print(
                        "The signals are too far apart with a lag of",
                        abs(dfn_2["X2"][j] - dfn_1["X1"][i]),
                        " s",
                    )

    ############################################################
    ############################################################

    # if ((snrp_L1>8 and snrp_H1>8) and (time_H1==time_L1)):
    #  print(Fore.BLUE + "There is a confirmed signal detection at {}s for both the detectors".format(time_L1))
    #  print(Style.RESET_ALL)
    # else:
    #  print("There was no simultaneous detection in H1 and L1 hence we rule out any authentic GW detection")
    #  print(Style.RESET_ALL)
    masses.append(x)
